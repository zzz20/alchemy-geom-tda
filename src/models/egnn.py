"""EGNN v17: протестированная конфигурация (loss 0.65 на 16 молекулах).

Ключевые параметры (найдены локальным тестированием):
  - update_coors=False (стабильно)
  - norm_feats=False (не уничтожает информацию)
  - m_dim=32 (больше ёмкость)
  - knn_graph_pytorch (без pyg-lib)
  - Нормализация координат pos / 5.0
  - nn.Embedding для типов атомов
  - Глобальные дескрипторы в heads
  - LayerNorm перед heads
  - lr=1e-4 (в train.py) — КРИТИЧНО! при 5e-4 не учится
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


class EGNNModel(nn.Module):
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

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,
                update_coors=False,
                update_feats=True,
                norm_feats=False,
                norm_coors=False,
                m_dim=32,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)

        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim

        # Нормализация глобальных дескрипторов (критично!)
        # Без этого mass~300 доминирует над mol_emb~1 после LayerNorm
        self.global_norm = nn.LayerNorm(global_dim)

        # ОТДЕЛЬНЫЕ heads для каждого таргета
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(head_in, hidden_channels), nn.SiLU(),
                nn.Linear(hidden_channels, 1))
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
        coors = batch.pos / 5.0  # нормализация координат

        edge_index = knn_graph(coors, k=16, batch=batch.batch, loop=False)
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)

        x = torch.cat([coors, feats], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
        h = x[:, 3:]

        mol_emb = global_add_pool(h, batch.batch)
        mol_emb = self.final_norm(mol_emb)
        global_desc = self._global_descriptors(batch)
        global_desc = self.global_norm(global_desc)  # Нормализация!
        mol_emb = torch.cat([mol_emb, global_desc], dim=-1)

        result = {}
        if self.predict_mu:
            result["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            result["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            result["gap"] = self.gap_head(mol_emb)
        return result


def build_egnn(predict_mu=True, predict_alpha=True, predict_gap=True, **kwargs):
    return EGNNModel(predict_mu=predict_mu, predict_alpha=predict_alpha,
                     predict_gap=predict_gap, **kwargs)
