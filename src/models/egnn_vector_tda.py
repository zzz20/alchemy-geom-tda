"""EGNN Vector + TDA: эквивариантный векторный диполь + TDA-фичи.

Объединение:
  - update_coors=True + norm_coors=True (для векторного выхода μ)
  - TDA-фичи конкатенируются с mol_emb перед скалярными heads (alpha, gap)
  - Векторный диполь: μ = Σ qᵢ·(rᵢ−COM)
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


class EGNNVectorTDA(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 5.0,
        tda_dim: int = 52,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not EGNN_AVAILABLE:
            raise ImportError("egnn-pytorch не установлен: pip install egnn-pytorch")

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.tda_dim = tda_dim
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        self.atom_embed = nn.Embedding(NUM_ATOM_TYPES, hidden_channels)

        # update_coors=True для векторного выхода, norm_coors=True для стабильности
        self.egnn_layers = nn.ModuleList([
            EGNN_Sparse(
                feats_dim=hidden_channels,
                pos_dim=3,
                edge_attr_dim=1,
                update_coors=True,
                update_feats=True,
                norm_feats=False,
                norm_coors=True,
                m_dim=32,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_channels)
        self.global_norm = nn.LayerNorm(NUM_ATOM_TYPES + 2)

        # Charge head для векторного диполя
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        # Скалярные heads с TDA
        global_dim = NUM_ATOM_TYPES + 2
        head_in = hidden_channels + global_dim + tda_dim

        if predict_alpha:
            self.alpha_skip = nn.Linear(global_dim, 1)
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
        coors = batch.pos / 5.0

        edge_index = knn_graph(coors, k=16, batch=batch.batch, loop=False)
        row, col = edge_index
        edge_dist = (coors[row] - coors[col]).norm(dim=-1, keepdim=True)

        x = torch.cat([coors, feats], dim=-1)
        for layer in self.egnn_layers:
            x = layer(x, edge_index, edge_attr=edge_dist, batch=batch.batch)
        updated_coors = x[:, :3]
        h = x[:, 3:]

        # === Эквивариантный диполь ===
        q = self.charge_head(h)
        mass = batch.x[:, -1:]
        weighted_coors = updated_coors * mass
        sum_weighted = global_add_pool(weighted_coors, batch.batch)
        sum_mass = global_add_pool(mass, batch.batch)
        com = sum_weighted / (sum_mass + 1e-8)
        shifted_coors = updated_coors - com[batch.batch]
        dipole_per_atom = q * shifted_coors
        mu = global_add_pool(dipole_per_atom, batch.batch)  # (B, 3)

        # === Скалярные выходы с TDA ===
        mol_emb = global_add_pool(h, batch.batch)
        mol_emb = self.final_norm(mol_emb)
        global_desc = self._global_descriptors(batch)
        global_desc = self.global_norm(global_desc)

        parts = [mol_emb, global_desc]
        if hasattr(batch, 'tda'):
            parts.append(batch.tda)
        mol_emb = torch.cat(parts, dim=-1)

        out = {"mu": mu}
        if self.predict_alpha:
            out["alpha"] = self.alpha_skip(global_desc) + self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_egnn_vector_tda(tda_dim=52, predict_alpha=True, predict_gap=True, **kwargs):
    return EGNNVectorTDA(tda_dim=tda_dim, predict_alpha=predict_alpha,
                          predict_gap=predict_gap, **kwargs)
