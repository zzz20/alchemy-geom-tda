"""Свой KNN graph — не требует pyg-lib.

Для каждого узла находит k ближайших соседей в его молекуле.
Работает на CPU и GPU, без внешних зависимостей кроме torch.
"""
import torch
from torch import Tensor


def knn_graph_pytorch(x: Tensor, k: int, batch: Tensor | None = None,
                      loop: bool = False) -> Tensor:
    """Построить kNN граф без pyg-lib.

    Args:
        x: (N, D) — координаты узлов
        k: число ближайших соседей
        batch: (N,) — индекс молекулы для каждого узла (если None — один граф)
        loop: включать ли self-loops

    Returns:
        edge_index: (2, E) — рёбра в формате PyG
    """
    N = x.shape[0]
    device = x.device

    # Считаем попарные расстояния
    # x: (N, D) → dist: (N, N)
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, D)
    dist = (diff ** 2).sum(-1)  # (N, N)

    # Маскируем узлы из других молекул (большие расстояния)
    if batch is not None:
        same_mol = (batch.unsqueeze(0) == batch.unsqueeze(1))  # (N, N)
        dist = dist.masked_fill(~same_mol, float('inf'))

    # Маскируем self-loops если нужно
    if not loop:
        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        dist = dist.masked_fill(diag_mask, float('inf'))

    # Для каждого узла находим k ближайших
    # k может быть больше числа узлов в молекуле — обрежем
    k_actual = min(k, N - 1 if not loop else N)
    if k_actual < 1:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    # topk: для каждой строки (узла) — k ближайших
    # indices: (N, k) — индексы соседей
    _, indices = dist.topk(k_actual, dim=-1, largest=False)  # (N, k)

    # Строим edge_index
    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k_actual)  # (N, k)
    edge_index = torch.stack([
        src.reshape(-1),    # source (N*k,)
        indices.reshape(-1)  # target (N*k,)
    ], dim=0)  # (2, N*k)

    # Удаляем рёбра с inf (когда соседей меньше k)
    valid = torch.isfinite(dist.gather(1, indices).reshape(-1))
    edge_index = edge_index[:, valid]

    return edge_index
