"""
PaiNN: Polarizable Atom Interaction Neural Network (Schütt et al., 2021)
https://arxiv.org/abs/2102.03150

Собственная реализация, не зависящая от версии PyTorch Geometric.

Особенности:
  - E(3)-эквивариантная (сдвиги + повороты + отражения + перестановки)
  - Узловые признаки раздельно: скаляры s_i (D,) и векторы v_i (D, 3)
  - Скалярные выходы mu, alpha, gap через pooling

Для Alchemy: предсказываем mu, alpha, gap (все скаляры, l=0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter


# ============================================================
# Bessel Basis Function (для радиальных расстояний)
# ============================================================
class BesselBasisLayer(nn.Module):
    """Радиальные Bessel basis functions (как в DimeNet/PaiNN)."""
    def __init__(self, num_radial: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.num_radial = num_radial
        # Обучаемые частоты
        self.freq = nn.Parameter(torch.arange(1, num_radial + 1, dtype=torch.float32) * torch.pi)
        # Обучаемый масштаб
        self.scale = nn.Parameter(torch.ones(num_radial))

    def forward(self, dist: Tensor) -> Tensor:
        """dist: (E,) → (E, num_radial)"""
        # Envelope function (smooth cutoff)
        envelope = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
        # Bessel basis
        bessel = torch.sin(self.freq * dist.unsqueeze(-1) / self.cutoff)
        return bessel * envelope.unsqueeze(-1) * self.scale


# ============================================================
# PaiNN Convolution Layer
# ============================================================
class PaiNNConv(nn.Module):
    """Один слой PaiNN: обновляет скалярные и векторные признаки."""

    def __init__(self, hidden_channels: int, num_radial: int = 16):
        super().__init__()
        self.hidden_channels = hidden_channels

        # 1. Message: из (s_i, s_j, v_i, v_j, r_ij) → (Δs_ij, Δv_ij)
        # Объединяем s_j и rbf → фильтр для s
        self.filter_s = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),  # для Δs и для gating
        )
        # Фильтр для v: использует rbf + direction
        self.filter_v = nn.Sequential(
            nn.Linear(num_radial, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),  # для Δv и для gating
        )

        # 2. Update: применяется к агрегированным сообщениям
        # Глубокий MLP для обновления s
        self.update_s = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
        )
        # Обновление v: через скалярный gating
        self.update_v = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),  # 3 компоненты для v
        )

    def forward(
        self,
        s: Tensor,           # (N, D) — скалярные признаки
        v: Tensor,           # (N, D, 3) — векторные признаки
        edge_index: Tensor,  # (2, E)
        edge_attr: Tensor,   # (E,) — расстояния
        vec_ij: Tensor,      # (E, 3) — единичный вектор от i к j, умноженный на rbf
    ) -> tuple[Tensor, Tensor]:
        """
        Возвращает обновлённые s и v.
        """
        N = s.shape[0]
        i, j = edge_index  # i — приёмник, j — источник

        # === Message ===
        # Bessel basis уже вычислены снаружи, передаём через edge_attr
        rbf = edge_attr  # (E, num_radial)

        # Фильтры для скаляров: (E, 2*D)
        fs = self.filter_s(rbf)
        # Δs_ij = fs[:, :D] * s_j + fs[:, D:] * (v_j · r_ij)
        # v_j: (N, D, 3) → берём для каждого edge: (E, D, 3)
        v_j = v[j]  # (E, D, 3)
        # v_j · vec_ij: (E, D, 3) * (E, 1, 3) → sum(-1) → (E, D)
        vj_dot_r = (v_j * vec_ij.unsqueeze(1)).sum(-1)  # (E, D)

        ds_msg = fs[:, :self.hidden_channels] * s[j] + \
                 fs[:, self.hidden_channels:] * vj_dot_r  # (E, D)

        # Фильтры для векторов: (E, 2*D)
        fv = self.filter_v(rbf)
        # Δv_ij = fv[:, :D] * v_j + fv[:, D:] * (s_j * r_ij)
        dv_msg = fv[:, :self.hidden_channels].unsqueeze(-1) * v_j + \
                 fv[:, self.hidden_channels:].unsqueeze(-1) * s[j].unsqueeze(-1) * vec_ij.unsqueeze(1)
        # (E, D, 3)

        # === Aggregate ===
        ds = scatter(ds_msg, i, dim=0, dim_size=N, reduce="sum")  # (N, D)
        dv = scatter(dv_msg, i, dim=0, dim_size=N, reduce="sum")  # (N, D, 3)

        # === Update ===
        # Конкатенируем старые s и агрегированное Δs
        s_cat = torch.cat([s, ds], dim=-1)  # (N, 2D)
        s_update = self.update_s(s_cat)  # (N, 2D)
        # s_new = s + a * ss, где a, ss из s_update
        a_s, ss = s_update.chunk(2, dim=-1)
        a_s = torch.sigmoid(a_s)
        s = s + a_s * ss

        # Для v: обновление через скалярный gating
        v_cat = torch.cat([s, ds], dim=-1)  # (N, 2D) — используем обновлённое s? Нет, по статье — оригинальное
        # Корректнее: используем s перед обновлением
        v_cat = torch.cat([s - a_s * ss, ds], dim=-1)  # возвращаемся к старому s
        v_update = self.update_v(v_cat)  # (N, 3D)
        a_v, vv, a_v2 = v_update.chunk(3, dim=-1)  # каждый (N, D)
        # v_new = v + a_v * vv * 1 + a_v2 * (v x r_ij) — упрощённо
        # Стандартное обновление: v_new = a_v * v + vv * dv (через gating)
        a_v = torch.sigmoid(a_v).unsqueeze(-1)  # (N, D, 1)
        v = a_v * v + vv.unsqueeze(-1) * dv  # (N, D, 3)

        return s, v


# ============================================================
# Главная модель PaiNN
# ============================================================
class PaiNNModel(nn.Module):
    """PaiNN для скалярных выходов (mu, alpha, gap).

    Args:
        hidden_channels: размер скрытых признаков (D)
        num_layers: число слоёв PaiNN
        num_rbf: число радиальных базисных функций
        cutoff: радиус обрезания (Å)
        predict_mu, predict_alpha, predict_gap: какие таргеты предсказывать
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов: 8 признаков → D
        self.atom_embed = nn.Linear(8, hidden_channels)

        # Инициализация векторных признаков нулевыми
        # v будет (N, D, 3), инициализируется нулями

        # Bessel basis для радиальных расстояний
        self.rbf = BesselBasisLayer(num_radial=num_rbf, cutoff=cutoff)

        # Стек слоёв PaiNN
        self.layers = nn.ModuleList([
            PaiNNConv(hidden_channels, num_rbf) for _ in range(num_layers)
        ])

        # Heads для скалярных выходов
        out_dim = hidden_channels
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.SiLU(),
                nn.Linear(out_dim // 2, 1),
            )
        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.SiLU(),
                nn.Linear(out_dim // 2, 1),
            )
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.SiLU(),
                nn.Linear(out_dim // 2, 1),
            )

    def forward(self, batch) -> dict[str, Tensor]:
        """
        Args:
            batch: PyG Batch с x (N, 8), pos (N, 3), batch (N,)
        """
        N = batch.x.shape[0]
        device = batch.x.device

        # Embedding
        s = self.atom_embed(batch.x.float())  # (N, D)
        v = torch.zeros(N, self.hidden_channels, 3, device=device)  # (N, D, 3)

        # Строим edge_index по радиусу (если нет в batch)
        if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
            edge_index = batch.edge_index
            # Считаем расстояния
            row, col = edge_index
            edge_vec = batch.pos[row] - batch.pos[col]  # (E, 3)
            edge_dist = edge_vec.norm(dim=-1)  # (E,)
        else:
            edge_index = radius_graph(batch.pos, r=self.cutoff, batch=batch.batch,
                                      loop=False, max_num_neighbors=32)
            row, col = edge_index
            edge_vec = batch.pos[row] - batch.pos[col]
            edge_dist = edge_vec.norm(dim=-1)

        # Маска по cutoff
        mask = edge_dist < self.cutoff
        edge_index = edge_index[:, mask]
        edge_vec = edge_vec[mask]
        edge_dist = edge_dist[mask]

        # Bessel basis
        rbf = self.rbf(edge_dist)  # (E, num_rbf)

        # Единичный вектор направления, умноженный на rbf (для векторных сообщений)
        # vec_ij: (E, 3), нормированный на 1
        edge_dir = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)  # (E, 3)

        # === Проходим через слои PaiNN ===
        for layer in self.layers:
            # Внутри layer используем rbf как edge_attr
            # vec_ij в статье = edge_dir * rbf.unsqueeze(-1) — но проще передать rbf и edge_dir отдельно
            # Адаптируем PaiNNConv: используем rbf как edge_attr, vec_ij = edge_dir
            s, v = layer(s, v, edge_index, rbf, edge_dir)

        # === Pooling ===
        mol_emb = scatter(s, batch.batch, dim=0, reduce="sum")  # (B, D)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)  # (B, 1)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)  # (B, 1)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)  # (B, 1)
        return out


def build_painn(
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> PaiNNModel:
    return PaiNNModel(
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
