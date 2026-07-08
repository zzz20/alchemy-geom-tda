"""TDA-фичи: персистентная гомология 3D облака атомов.

Для каждой молекулы:
  1. Строим Vietoris-Rips фильтрацию на 3D координатах атомов
  2. Считаем персистентность для H_0, H_1, H_2
  3. Извлекаем фичи:
     - Betti curves (гистограмма числа особенностей по уровням фильтрации)
     - Persistence entropy (энтропия распределения персистентностей)
     - Средняя персистентность
     - Число "длинных" особенностей ( персистентность > 0.5 Å)
  4. Конкатенируем в вектор фиксированной размерности

Эти фичи E(3)-инвариантны (топология не меняется при изометриях).
"""
import numpy as np
import torch
from torch import Tensor

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False


def compute_persistence(coords: np.ndarray, max_dim: int = 2, max_radius: float = 5.0) -> list[list[tuple[float, float]]]:
    """Построить Vietoris-Rips комплекс и вычислить персистентность.

    Args:
        coords: (N, 3) координаты атомов
        max_dim: максимальная размерность гомологии (0, 1, 2)
        max_radius: максимальный радиус фильтрации (Å)

    Returns:
        persistence: list[ [(birth, death), ...], ... ] для каждой размерности 0..max_dim
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("GUDHI не установлен. pip install gudhi")

    n_atoms = len(coords)
    if n_atoms < 2:
        return [[] for _ in range(max_dim + 1)]

    # Используем RipsComplex из GUDHI
    rips = gd.RipsComplex(points=coords.tolist(), max_edge_length=max_radius)
    st = rips.create_simplex_tree(max_dimension=max_dim + 1)
    persistence = st.persistence(homology_coeff_field=2, min_persistence=0.0)

    # Группируем по размерности
    by_dim = [[] for _ in range(max_dim + 1)]
    for dim, (birth, death) in persistence:
        if dim <= max_dim:
            by_dim[dim].append((birth, death))
    return by_dim


def betti_curve(persistence_pairs: list[tuple[float, float]], n_bins: int = 32, max_r: float = 5.0) -> np.ndarray:
    """Вычислить Betti curve — число активных особенностей по уровням фильтрации.

    Для каждого уровня r ∈ [0, max_r] считаем сколько особенностей (birth ≤ r < death).
    Дискретизуем по n_bins точкам.

    Returns:
        (n_bins,) массив
    """
    bins = np.linspace(0, max_r, n_bins)
    curve = np.zeros(n_bins, dtype=np.float32)
    for birth, death in persistence_pairs:
        if death == float("inf"):
            death = max_r + 1
        for i, r in enumerate(bins):
            if birth <= r < death:
                curve[i] += 1
    return curve


def persistence_entropy(persistence_pairs: list[tuple[float, float]], eps: float = 1e-10) -> float:
    """Persistence entropy: -Σ p_i * log(p_i), где p_i = persistence_i / sum(persistence)."""
    if not persistence_pairs:
        return 0.0
    persistences = np.array([d - b for b, d in persistence_pairs if d != float("inf")])
    if len(persistences) == 0 or persistences.sum() < eps:
        return 0.0
    p = persistences / persistences.sum()
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def extract_tda_features(
    coords: np.ndarray,
    n_bins: int = 16,
    max_radius: float = 5.0,
    max_dim: int = 2,
) -> np.ndarray:
    """Извлечь TDA-фичи из 3D координат атомов молекулы.

    Returns:
        np.ndarray размерности n_bins * (max_dim + 1) + (max_dim + 1) + 1
        = 16*3 + 3 + 1 = 52 (по умолчанию)
    """
    if not GUDHI_AVAILABLE:
        # Если gudhi нет, возвращаем нули
        return np.zeros(n_bins * (max_dim + 1) + (max_dim + 1) + 1, dtype=np.float32)

    persistence = compute_persistence(coords, max_dim=max_dim, max_radius=max_radius)

    features = []
    for dim in range(max_dim + 1):
        # Betti curve
        betti = betti_curve(persistence[dim], n_bins=n_bins, max_r=max_radius)
        features.append(betti)
        # Persistence entropy
        features.append(np.array([persistence_entropy(persistence[dim])], dtype=np.float32))

    # Средняя персистентность по всем размерностям
    all_persist = []
    for dim_pairs in persistence:
        for b, d in dim_pairs:
            if d != float("inf"):
                all_persist.append(d - b)
    mean_persist = float(np.mean(all_persist)) if all_persist else 0.0
    features.append(np.array([mean_persist], dtype=np.float32))

    return np.concatenate(features)


def extract_tda_features_batch(
    coords_batch: list[np.ndarray],
    n_bins: int = 16,
    max_radius: float = 5.0,
    max_dim: int = 2,
) -> np.ndarray:
    """Извлечь TDA-фичи для батча молекул.

    Args:
        coords_batch: list of (N_i, 3) массивов

    Returns:
        (B, F) массив — F = n_bins * (max_dim+1) + (max_dim+1) + 1
    """
    return np.stack([
        extract_tda_features(c, n_bins=n_bins, max_radius=max_radius, max_dim=max_dim)
        for c in coords_batch
    ])


def tda_feature_dim(n_bins: int = 16, max_dim: int = 2) -> int:
    """Размерность TDA-фичей."""
    return n_bins * (max_dim + 1) + (max_dim + 1) + 1


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    coords = np.random.randn(10, 3).astype(np.float32)
    feats = extract_tda_features(coords)
    print(f"Координаты: {coords.shape}")
    print(f"TDA-фичи: {feats.shape} (ожидаемая размерность: {tda_feature_dim()})")
    print(f"Первые 10 значений: {feats[:10]}")
