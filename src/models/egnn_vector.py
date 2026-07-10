"""EGNN с векторным выходом для дипольного момента μ (l=1).

Вместо предсказания скалярного |μ|, предсказываем вектор μ ∈ R³.
Это эквивариантный выход (l=1) — при повороте молекулы вектор поворачивается вместе с ней.

Архитектура:
  1. EGNN слои обновляют координаты атомов (update_coors=True)
  2. Сдвиг центра масс: μ = Σ_i q_i * (pos_i - COM), где q_i — выучиваемый заряд атома
  3. Это СТРОГО эквивариантно: при повороте pos_i → R·pos_i, μ → R·μ

Дополнительно предсказываем скаляры alpha и gap через отдельный head.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool

try:
    from egnn_pytorch import EGNN_Sparse
    EGNN_AVAILABLE = True
except ImportError:
    EGNN_AVAILABLE = False

from .knn import knn_graph_pytorch as knn_graph

NUM_ATOM_TYPES = 7


class EGNNVectorModel(nn.Module):
    """EGNN с эквивариантным векторным выходом для диполя μ.

    Диполь: μ = Σ_i q_i * (r_i - COM)
    где q_i — выучиваемый заряд атома типа i, r_i — позиция атома, COM — центр масс.

    Это СТРОГО эквивариантно:
    - При сдвиге: r_i → r_i + t, COM → COM + t, (r_i - COM) не меняется → μ не меняется ✓
    - При повороте: r_i → R·r_i, COM → R·COM, (r_i - COM) → R·(r_i - COM) → μ → R·μ ✓
    - При перестановке: сумма не меняется ✓

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв EGNN
        predict_alpha: предсказывать скалярную alpha
        predict_gap: предсказывать скалярный gap
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError("egnn-pytorch не установлен: pip install egnn-pytorch")

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        # EGNN слои — update_coors=True (обновляем координаты для эквивариантности)
        # НО с norm_coors=True для стабильности
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,
                update_coors=True,      # обновляем координаты!
                update_feats=True,
                norm_feats=False,
                norm_coors=True,        # нормализация координат для стабильности
                m_dim=32,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)

        # Charge head: предсказывает заряд q_i для каждого атома из его признаков
        # q_i ∈ R (скаляр) — это будет вес для позиции атома
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Скалярные heads для alpha и gap
        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim

        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))

    def _global_descriptors(self, batch) -> Tensor:
        atom_onehot = batch.x[:, :NUM_ATOM_TYPES]
        mass = batch.x[:, -1:]
        hist = global_add_pool(atom_onehot, batch.batch)
        ones = torch.ones(mass.shape[0], 1, device=mass.device)
        n_atoms = global_add_pool(ones, batch.batch)
        total_mass = global_add_pool(mass, batch.batch)
        return torch.cat([hist, n_atoms, total_mass], dim=-1)

    def forward(self, batch) -> dict[str, Tensor]:
        atom_types = batch.x[:, :NUM_ATOM_TYPES].argmax(dim=-1).long()
        feats = self.atom_embed(atom_types)
        coors = batch.pos / 5.0  # нормализация для стабильности

        edge_index = knn_graph(coors, k=16, batch=batch.batch, loop=False)
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)

        x = torch.cat([coors, feats], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
        # x: (N, 3 + hidden) — обновлённые координаты + признаки
        updated_coors = x[:, :3]  # (N, 3) — обновлённые позиции
        h = x[:, 3:]  # (N, hidden) — признаки

        # === Эквивариантный диполь ===
        # q_i = charge_head(h_i) — заряд атома
        q = self.charge_head(h)  # (N, 1)

        # COM для каждой молекулы (центр масс обновлённых координат)
        # Используем массу атома как вес
        mass = batch.x[:, -1:]  # (N, 1)
        # COM = Σ(mass_i * coors_i) / Σ(mass_i)
        weighted_coors = updated_coors * mass  # (N, 3)
        sum_weighted = global_add_pool(weighted_coors, batch.batch)  # (B, 3)
        sum_mass = global_add_pool(mass, batch.batch)  # (B, 1)
        com = sum_weighted / (sum_mass + 1e-8)  # (B, 3)

        # Сдвинутые координаты: r_i - COM (для каждого атома своей молекулы)
        # com[batch.batch]: (N, 3) — COM для каждого атома
        shifted_coors = updated_coors - com[batch.batch]  # (N, 3)

        # Диполь: μ = Σ_i q_i * (r_i - COM)
        # q_i * shifted_coors_i: (N, 1) * (N, 3) → (N, 3)
        dipole_per_atom = q * shifted_coors  # (N, 3)
        mu = global_add_pool(dipole_per_atom, batch.batch)  # (B, 3) — ВЕКТОР!

        # === Скалярные выходы ===
        mol_emb = global_add_pool(h, batch.batch)
        mol_emb = self.final_norm(mol_emb)
        global_desc = self._global_descriptors(batch)
        mol_emb = torch.cat([mol_emb, global_desc], dim=-1)

        out = {"mu": mu}  # (B, 3) — вектор!
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)  # (B, 1)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)  # (B, 1)
        return out


def build_egnn_vector(predict_alpha=True, predict_gap=True, **kwargs):
    return EGNNVectorModel(predict_alpha=predict_alpha, predict_gap=predict_gap, **kwargs)
