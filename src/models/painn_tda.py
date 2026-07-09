"""PaiNN + TDA: основная модель проекта."""
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter

from .painn import PaiNNModel, PaiNNInteraction, PaiNNMix, BesselBasisLayer, DistanceEnvelope
from ..tda.film import FiLMNodeModulation


class PaiNNTDA(PaiNNModel):
    """PaiNN с FiLM conditioning на TDA-фичах."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        predict_mu: bool = True,
        predict_alpha: bool = True,
        predict_gap: bool = True,
        tda_dim: int = 52,
        tda_film_position: int | None = None,
        max_num_neighbors: int = 32,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            predict_mu=predict_mu,
            predict_alpha=predict_alpha,
            predict_gap=predict_gap,
            max_num_neighbors=max_num_neighbors,
        )
        if tda_film_position is None:
            tda_film_position = num_layers // 2
        self.tda_film_position = tda_film_position
        self.film = FiLMNodeModulation(tda_dim, hidden_channels)

    def forward(self, batch) -> dict:
        N = batch.x.shape[0]
        device = batch.x.device

        s = self.atom_embed(batch.x.float())
        v = torch.zeros(N, self.hidden_channels, 3, device=device)

        edge_index = radius_graph(
            batch.pos, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_vec = batch.pos[row] - batch.pos[col]
        edge_dist = edge_vec.norm(dim=-1)

        mask = edge_dist < self.cutoff
        edge_index = edge_index[:, mask]
        edge_vec = edge_vec[mask]
        edge_dist = edge_dist[mask]

        rbf = self.rbf(edge_dist)
        edge_weight = self.envelope(edge_dist)
        vec_ij = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)

        for i, (interaction, mix) in enumerate(zip(self.interactions, self.mixes)):
            ds, dvec = interaction(s, v, edge_index, rbf, edge_weight, vec_ij)
            s, v = mix(s, v, ds, dvec)
            # FiLM после указанного слоя
            if i == self.tda_film_position - 1 and hasattr(batch, 'tda'):
                tda = batch.tda
                s = self.film(s, tda, batch.batch)

        mol_emb = scatter(s, batch.batch, dim=0, reduce='sum')

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
        return out


def build_painn_tda(
    tda_dim: int = 52,
    predict_mu: bool = True,
    predict_alpha: bool = True,
    predict_gap: bool = True,
    **kwargs,
) -> PaiNNTDA:
    return PaiNNTDA(
        tda_dim=tda_dim,
        predict_mu=predict_mu,
        predict_alpha=predict_alpha,
        predict_gap=predict_gap,
        **kwargs,
    )
