"""EGNN v9: с Embedding для типов атомов и глобальными дескрипторами.

Главные отличия от v8:
  1. atom_embed = nn.Embedding(7, hidden) — тип атома как индекс (а не Linear)
  2. Глобальные дескрипторы молекулы (гистограмма типов + число атомов + масса)
     подаются прямо в heads, минуя EGNN. Это даёт модели тривиальный бейзлайн:
     "alpha ≈ 5*n_C + 3*n_O + ..." — она сможет это выучить.
  3. radius_graph (радиус 5 Å) — все пары атомов в радиусе
  4. Правильный API EGNN_Sparse: feats и pos отдельно
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
    """EGNN с Embedding и глобальными дескрипторами."""

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

        # Embedding атомов: индекс типа → hidden (вместо Linear)
        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        # EGNN слои
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_dim=1,
            )
            for _ in range(num_layers)
        ])

        # Размер глобальных дескрипторов: 7 (гистограмма) + 1 (n_atoms) + 1 (mass)
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
        """Вычислить глобальные дескрипторы молекулы.

        Возвращает (B, 9): [hist_C, hist_N, ..., hist_Cl, n_atoms, total_mass]
        """
        # x: (N, 8) — one-hot (7) + mass (1)
        atom_onehot = batch.x[:, :NUM_ATOM_TYPES]  # (N, 7)
        mass = batch.x[:, -1:]  # (N, 1)

        # Гистограмма: сумма one-hot по узлам каждой молекулы
        hist = global_add_pool(atom_onehot, batch.batch)  # (B, 7)
        n_atoms = global_add_pool(torch.ones(mass.shape[0], 1, device=mass.device), batch.batch)
        total_mass = global_add_pool(mass, batch.batch)  # (B, 1)

        return torch.cat([hist, n_atoms, total_mass], dim=-1)  # (B, 9)

    def forward(self, batch) -> dict[str, Tensor]:
        # Извлекаем индекс типа атома из one-hot
        atom_types = batch.x[:, :NUM_ATOM_TYPES].argmax(dim=-1).long()  # (N,)

        # Embedding
        h = self.atom_embed(atom_types)  # (N, hidden)
        pos = batch.pos  # (N, 3)

        # radius_graph
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=64,
        )
        row, col = edge_index
        edge_dist = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)  # (E, 1)

        # EGNN слои
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index, edge_attr=edge_dist)

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
