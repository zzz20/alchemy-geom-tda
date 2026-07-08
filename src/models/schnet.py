"""SchNet baseline: учитывает сдвиги и перестановки, но не повороты.

Использует готовую реализацию из PyTorch Geometric.
Учитывает только межатомные расстояния (инвариантные к вращениям), поэтому
выход — скаляр. Для диполя и поляризуемости используем hack: предсказываем
компоненты как скаляры (теряем эквивариантность).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SchNet


class SchNetWrapper(nn.Module):
    """SchNet + линейный head для произвольного числа выходов."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        out_dim: int = 3,  # 3 для диполя
        readout: str = "mean",
    ):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )
        # Head: из hidden_channels в out_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch с x, pos, batch
        Returns:
            (B, out_dim)
        """
        # SchNet возвращает (B, hidden_channels) после readout
        emb = self.schnet(batch.x, batch.pos, batch.batch)
        return self.head(emb)


def build_schnet(out_dim: int = 3, **kwargs) -> SchNetWrapper:
    return SchNetWrapper(out_dim=out_dim, **kwargs)
