"""
PaiNN: Polarizable Atom Interaction Neural Network (Schütt et al., 2021)
https://arxiv.org/abs/2102.03150

Чистая реализация по образцу torchmd-net (https://github.com/torchmd/torchmd-net).
E(3)-эквивариантная: сдвиги + повороты + отражения + перестановки.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter


class DistanceEnvelope(nn.Module):
    """Smooth envelope for distance features (as in DimeNet)."""
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        # 0.5 * (cos(pi * d / cutoff) + 1), 0 outside cutoff
        env = 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0)
        return env * (dist < self.cutoff).float()


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = DistanceEnvelope(cutoff)
        self.freq = nn.Parameter(torch.arange(1, num_radial + 1, dtype=torch.float32) * torch.pi)

    def forward(self, dist):
        # dist: (E,)
        env = self.envelope(dist)  # (E,)
        bessel = torch.sin(self.freq * dist.unsqueeze(-1) / self.cutoff)  # (E, num_radial)
        return bessel * env.unsqueeze(-1)


class PaiNNInteraction(nn.Module):
    """Message passing: вычисляет Δs и Δv по рёбрам."""
    def __init__(self, hidden_channels: int, num_rbf: int, activation=nn.SiLU):
        super().__init__()
        self.hidden_channels = hidden_channels
        # Фильтр рёбер: rbf → 2*D (для s и v частей)
        self.filter_layers = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.filter = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )

    def forward(self, s, v, edge_index, rbf, edge_weight, vec_ij):
        """
        s: (N, D), v: (N, D, 3)
        rbf: (E, num_rbf)
        edge_weight: (E,) — envelope (smooth cutoff)
        vec_ij: (E, 3) — единичный вектор направления r_ij / |r_ij|
        """
        j, i = edge_index

        # Фильтр рёбер
        f = self.filter_layers(rbf)  # (E, D)
        f = f * edge_weight.unsqueeze(-1)  # (E, D) — маска по cutoff
        f = self.filter(f)  # (E, 2D)
        f1, f2 = f.chunk(2, dim=-1)  # каждый (E, D)

        # Признаки соседей
        s_j = s[j]  # (E, D)
        v_j = v[j]  # (E, D, 3)

        # Скалярное сообщение: f1 * s_j
        ds = f1 * s_j  # (E, D)

        # Векторное сообщение: f2 * v_j + (f1 * s_j) * vec_ij
        # f2: (E, D) → (E, D, 1)
        # v_j: (E, D, 3)
        # (f1 * s_j): (E, D) → (E, D, 1) * vec_ij: (E, 1, 3) → (E, D, 3)
        dvec = f2.unsqueeze(-1) * v_j + (f1 * s_j).unsqueeze(-1) * vec_ij.unsqueeze(1)
        # dvec: (E, D, 3)

        # Агрегация: суммирование по соседям
        ds = scatter(ds, i, dim=0, dim_size=s.shape[0], reduce='sum')  # (N, D)
        dvec = scatter(dvec, i, dim=0, dim_size=v.shape[0], reduce='sum')  # (N, D, 3)

        return ds, dvec


class PaiNNMix(nn.Module):
    """Update: обновляет s и v с помощью агрегированных сообщений."""
    def __init__(self, hidden_channels: int, activation=nn.SiLU):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            activation(),
            nn.Linear(2 * hidden_channels, 3 * hidden_channels),
        )

    def forward(self, s, v, ds, dvec):
        """
        s: (N, D), v: (N, D, 3), ds: (N, D), dvec: (N, D, 3)
        """
        # Конкатенируем s и ds
        s_cat = torch.cat([s, ds], dim=-1)  # (N, 2D)
        s_update = self.update(s_cat)  # (N, 3D)
        a, ss, vsv = s_update.chunk(3, dim=-1)  # каждый (N, D)

        # Скалярное обновление: residual + gating
        s = s + torch.sigmoid(a) * ss  # (N, D)

        # Векторное обновление: residual + gating
        # v' = v + sigmoid(vsv) * dvec
        v = v + torch.sigmoid(vsv).unsqueeze(-1) * dvec  # (N, D, 3)

        return s, v


class PaiNNModel(nn.Module):
    """PaiNN для скалярных выходов (mu, alpha, gap).

    Args:
        hidden_channels: размер скрытых признаков
        num_layers: число слоёв
        num_rbf: число радиальных базисных функций
        cutoff: радиус обрезания (Å)
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
        max_num_neighbors: int = 32,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.predict_mu = predict_mu
        self.predict_alpha = predict_alpha
        self.predict_gap = predict_gap

        # Embedding атомов
        self.atom_embed = nn.Linear(8, hidden_channels)

        # Bessel basis + envelope
        self.rbf = BesselBasisLayer(num_radial=num_rbf, cutoff=cutoff)
        self.envelope = DistanceEnvelope(cutoff)

        # Слои PaiNN: interaction + mix
        self.interactions = nn.ModuleList([
            PaiNNInteraction(hidden_channels, num_rbf) for _ in range(num_layers)
        ])
        self.mixes = nn.ModuleList([
            PaiNNMix(hidden_channels) for _ in range(num_layers)
        ])

        # Heads для скалярных выходов
        if predict_mu:
            self.mu_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
        if predict_alpha:
            self.alpha_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
        if predict_gap:
            self.gap_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )

    def forward(self, batch) -> dict:
        N = batch.x.shape[0]
        device = batch.x.device

        # Embedding
        s = self.atom_embed(batch.x.float())  # (N, D)
        v = torch.zeros(N, self.hidden_channels, 3, device=device)  # (N, D, 3)

        # Строим рёбра по радиусу
        edge_index = radius_graph(
            batch.pos, r=self.cutoff, batch=batch.batch,
            loop=False, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_vec = batch.pos[row] - batch.pos[col]  # (E, 3)
        edge_dist = edge_vec.norm(dim=-1)  # (E,)

        # Маска по cutoff (на всякий случай)
        mask = edge_dist < self.cutoff
        edge_index = edge_index[:, mask]
        edge_vec = edge_vec[mask]
        edge_dist = edge_dist[mask]

        # Bessel basis и envelope
        rbf = self.rbf(edge_dist)  # (E, num_rbf)
        edge_weight = self.envelope(edge_dist)  # (E,)

        # Единичный вектор направления
        vec_ij = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)  # (E, 3)

        # Проходим через слои
        for interaction, mix in zip(self.interactions, self.mixes):
            ds, dvec = interaction(s, v, edge_index, rbf, edge_weight, vec_ij)
            s, v = mix(s, v, ds, dvec)

        # Pooling: суммарный вектор молекулы из скалярных признаков
        mol_emb = scatter(s, batch.batch, dim=0, reduce='sum')  # (B, D)

        out = {}
        if self.predict_mu:
            out["mu"] = self.mu_head(mol_emb)
        if self.predict_alpha:
            out["alpha"] = self.alpha_head(mol_emb)
        if self.predict_gap:
            out["gap"] = self.gap_head(mol_emb)
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
