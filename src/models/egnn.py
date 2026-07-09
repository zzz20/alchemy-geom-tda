"""EGNN v10: КОРРЕКТНЫЙ API egnn-pytorch.

Главные исправления относительно v9:
  1. EGNN_Sparse принимает СКЛЕЕННЫЙ тензор [pos, feats], не два отдельных
  2. EGNN_Sparse возвращает СКЛЕЕННЫЙ тензор [coors_out, hidden_out]
  3. Параметр называется edge_attr_dim, а не edge_dim
  4. Передаём batch в forward (нужен для norm_feats)
  5. Embedding для типов атомов
  6. Глобальные дескрипторы в heads (гистограмма + n_atoms + mass)
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, radius_graph

try:
    from egnn_pytorch import EGNN_Sparse
    EGNN_AVAILABLE = True
except ImportError:
    EGNN_AVAILABLE = False

NUM_ATOM_TYPES = 7  # H, C, N, O, F, S, Cl


class EGNNModel(nn.Module):
    """EGNN для скалярных выходов (mu, alpha, gap).

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв EGNN
        cutoff: радиус для radius_graph (Å)
        predict_mu, predict_alpha, predict_gap: какие таргеты предсказывать
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError("egnn-pytorch не установлен: pip install egnn-pytorch")

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        # EGNN слои — КОРРЕКТНЫЕ ИМЕНА ПАРАМЕТРОВ
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,       # ← НЕ edge_dim!
                update_coors=True,     # обновляем координаты (эквивариантность)
                update_feats=True,     # обновляем признаки
                norm_feats=True,       # LayerNorm признаков (нужен batch)
                norm_coors=False,      # без нормализации координат
            )
            for _ in range(num_layers)
        ])

        # Глобальные дескрипторы: 7 (hist) + 1 (n_atoms) + 1 (mass) = 9
        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim

        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )

    def _global_descriptors(self, batch) -> Tensor:
        """Гистограмма типов атомов + n_atoms + total_mass. (B, 9)"""
        atom_onehot = batch.x[:, :NUM_ATOM_TYPES]  # (N, 7)
        mass = batch.x[:, -1:]                      # (N, 1)

        hist = global_add_pool(atom_onehot, batch.batch)  # (B, 7)
        ones = torch.ones(mass.shape[0], 1, device=mass.device)
        n_atoms = global_add_pool(ones, batch.batch)       # (B, 1)
        total_mass = global_add_pool(mass, batch.batch)    # (B, 1)

        return torch.cat([hist, n_atoms, total_mass], dim=-1)  # (B, 9)

    def forward(self, batch) -> dict[str, Tensor]:
        # Индекс типа атома из one-hot
        atom_types = batch.x[:, :NUM_ATOM_TYPES].argmax(dim=-1).long()  # (N,)

        # Embedding
        feats = self.atom_embed(atom_types)  # (N, hidden)
        coors = batch.pos                    # (N, 3)

        # radius_graph — все пары атомов в радиусе cutoff
        edge_index = radius_graph(
            coors, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=64,
        )
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)  # (E, 1)

        # === КОРРЕКТНЫЙ ВЫЗОВ EGNN_Sparse ===
        # x = склеенный [pos, feats]
        x = torch.cat([coors, feats], dim=-1)  # (N, 3 + hidden)

        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
            # x: (N, 3 + hidden) — склеенный [coors_out, hidden_out]

        # Разделяем обратно
        # coors_out = x[:, :3]  # не используем для скалярных выходов
        h = x[:, 3:]  # (N, hidden)

        # Pooling
        mol_emb = global_add_pool(h, batch.batch)  # (B, hidden)

        # Глобальные дескрипторы
        global_desc = self._global_descriptors(batch)  # (B, 9)

        # Конкатенируем
        mol_emb = torch.cat([mol_emb, global_desc], dim=-1)  # (B, hidden + 9)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_egnn(
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> EGNNModel:
    return EGNNModel(
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
