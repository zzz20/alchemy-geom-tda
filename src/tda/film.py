"""FiLM conditioning: модуляция признаков TDA-фичами.

FiLM (Feature-wise Linear Modulation):
  γ, β = MLP(tda_features)
  features' = γ * features + β

TDA-фичи E(3)-инвариантны, поэтому модуляция не нарушает эквивариантность:
  - Если модулируем скалярные признаки → остаётся скаляр (l=0)
  - Если модулируем векторные признаки → остаётся вектор (l=1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation.

    Args:
        tda_dim: размерность TDA-фичей
        feat_dim: размерность модулируемых признаков
        hidden_dim: размерность MLP
    """

    def __init__(self, tda_dim: int, feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.tda_dim = tda_dim
        self.feat_dim = feat_dim

        # MLP: TDA → (γ, β) каждый размерности feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(tda_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * feat_dim),
        )
        # Инициализация: γ=1, β=0 в начале обучения
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:feat_dim] = 1.0  # γ = 1

    def forward(self, features: Tensor, tda: Tensor) -> Tensor:
        """
        Args:
            features: (B, feat_dim) или (N, feat_dim) — модулируемые признаки
            tda: (B, tda_dim) — TDA-фичи на уровне молекулы

        Returns:
            модифицированные признаки той же формы
        """
        # γ, β: (B, 2 * feat_dim)
        film = self.mlp(tda)
        gamma, beta = film.chunk(2, dim=-1)  # каждый (B, feat_dim)

        if features.dim() == 2 and features.shape[0] == gamma.shape[0]:
            # (B, feat_dim) — прямой случай
            return gamma * features + beta
        elif features.dim() == 2:
            # (N, feat_dim) — нужно расширить по узлам
            # Предполагаем, что в батче есть индексы узлов
            # Эта ветка не используется напрямую — см. FiLMNodeModulation
            raise ValueError("Используйте FiLMNodeModulation для узловых признаков")
        else:
            raise ValueError(f"Неподдерживаемая форма features: {features.shape}")


class FiLMNodeModulation(nn.Module):
    """FiLM для узловых признаков с учётом принадлежности к молекуле.

    TDA-фичи заданы на уровне молекулы, модулируют узловые признаки.
    """

    def __init__(self, tda_dim: int, feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.film = FiLMModulation(tda_dim, feat_dim, hidden_dim)

    def forward(self, node_features: Tensor, tda: Tensor, batch_idx: Tensor) -> Tensor:
        """
        Args:
            node_features: (N, feat_dim) — узловые признаки
            tda: (B, tda_dim) — TDA-фичи молекул
            batch_idx: (N,) — индекс молекулы для каждого узла

        Returns:
            (N, feat_dim) — модулированные признаки
        """
        # Расширяем TDA-фичи до каждого узла
        tda_per_node = tda[batch_idx]  # (N, tda_dim)

        # γ, β: (N, 2 * feat_dim)
        film = self.film.mlp(tda_per_node)
        gamma, beta = film.chunk(2, dim=-1)  # (N, feat_dim) каждый

        return gamma * node_features + beta
