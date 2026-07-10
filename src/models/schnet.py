"""SchNet с собственным interaction_graph (не требует pyg-lib).

Использует knn_graph_pytorch вместо RadiusInteractionGraph.
"""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import SchNet

from .knn import knn_graph_pytorch


class KNNInteractionGraph:
    """Заменяет RadiusInteractionGraph — не требует pyg-lib."""
    def __init__(self, k: int = 16):
        self.k = k

    def __call__(self, pos: Tensor, batch: Tensor):
        edge_index = knn_graph_pytorch(pos, k=self.k, batch=batch, loop=False)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class SchNetWrapper(nn.Module):
    """SchNet + линейный head. Не требует pyg-lib."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        out_dim: int = 3,
        readout: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
            interaction_graph=KNNInteractionGraph(k=16),
        )
        # SchNet имеет lin1 и lin2 — убираем их, чтобы получить raw embeddings
        self.schnet.lin1 = nn.Identity()
        self.schnet.lin2 = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, batch) -> Tensor:
        atom_types = batch.x.argmax(dim=-1).long()
        emb = self.schnet(atom_types, batch.pos, batch.batch)
        return self.head(emb)


def build_schnet(out_dim: int = 3, **kwargs) -> SchNetWrapper:
    return SchNetWrapper(out_dim=out_dim, **kwargs)
