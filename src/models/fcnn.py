"""FCNN baseline: полносвязная сеть на фичах молекулы.

Никаких индуктивных смещений: ни сдвигов, ни поворотов, ни перестановок.
Фичи молекулы: агрегированные узловые признаки (mean, max, sum) + глобальные статистики.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNNBaseline(nn.Module):
    """Простая MLP регрессия.

    Вход: вектор признаков молекулы (агрегированные узловые признаки).
    Выход: предсказание таргета.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        out_dim: int = 3,  # 3 для диполя, 6 для α (hack), 1 для gap
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: PyG Batch с полями x (N, F), batch (N,)
        Returns:
            (B, out_dim)
        """
        from torch_geometric.utils import scatter

        x = batch.x  # (N, F)
        batch_idx = batch.batch  # (N,)

        # Глобальные признаки молекулы: mean, max, sum по узлам
        mean = scatter(x, batch_idx, dim=0, reduce="mean")  # (B, F)
        mx = scatter(x, batch_idx, dim=0, reduce="max")     # (B, F)
        s = scatter(x, batch_idx, dim=0, reduce="sum")      # (B, F)
        feat = torch.cat([mean, mx, s], dim=-1)              # (B, 3F)

        return self.net(feat)


def build_fcnn(in_dim: int, out_dim: int, **kwargs) -> FCNNBaseline:
    return FCNNBaseline(in_dim=in_dim, out_dim=out_dim, **kwargs)
