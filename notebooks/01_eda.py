# EDA: исследование датасета Alchemy
"""
Ноутбук для исследования датасета Alchemy.

Запуск:
  jupyter notebook notebooks/01_eda.py
  или
  python notebooks/01_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import parse_alchemy_sdf, ATOM_TYPES

# Стиль графиков
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_sample_molecules(data_dir: str = "data/alchemy/data", n_max: int = 1000):
    """Загрузить первые n_max молекул для EDA."""
    raw_dir = Path(data_dir)
    sdf_files = sorted(raw_dir.glob("*.sdf"))
    if not sdf_files:
        raise FileNotFoundError(f"SDF не найдены в {raw_dir}")

    print(f"Найдено {len(sdf_files)} SDF файлов. Загружаю {sdf_files[0].name}...")
    molecules = parse_alchemy_sdf(str(sdf_files[0]))
    print(f"Загружено {len(molecules)} молекул из {sdf_files[0].name}")

    if len(molecules) > n_max:
        molecules = molecules[:n_max]
        print(f"Ограничился первыми {n_max} молекулами")
    return molecules


def analyze_atom_distribution(molecules):
    """Распределение типов атомов."""
    print("\n=== Распределение типов атомов ===")
    atom_counts = {a: 0 for a in ATOM_TYPES}
    sizes = []

    for mol in molecules:
        sizes.append(len(mol["atoms"]))
        for symbol, *_ in mol["atoms"]:
            if symbol in atom_counts:
                atom_counts[symbol] += 1

    print("Атомы:")
    for a, c in sorted(atom_counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c}")

    print(f"\nРазмер молекул: mean={np.mean(sizes):.1f}, "
          f"min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.1f}")

    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(atom_counts.keys(), atom_counts.values())
    axes[0].set_title("Распределение типов атомов")
    axes[0].set_ylabel("Количество")

    axes[1].hist(sizes, bins=30, edgecolor="black")
    axes[1].set_title("Распределение размеров молекул")
    axes[1].set_xlabel("Число атомов")
    axes[1].set_ylabel("Частота")

    plt.tight_layout()
    plt.savefig("results/figures/eda_atoms.png", dpi=100, bbox_inches="tight")
    plt.show()


def analyze_targets(molecules):
    """Распределение целевых свойств."""
    print("\n=== Целевые свойства ===")

    dipoles = []        # нормы диполя
    polarizabilities = []  # след тензора поляризуемости
    gaps = []

    for mol in molecules:
        props = mol["properties"]
        if "dipole_moment" in props:
            try:
                mu = np.array([float(v) for v in props["dipole_moment"].split()])
                if len(mu) == 3:
                    dipoles.append(np.linalg.norm(mu))
                elif len(mu) == 1:
                    dipoles.append(mu[0])
            except (ValueError, AttributeError):
                pass

        if "polarizability" in props:
            try:
                alpha = np.array([float(v) for v in props["polarizability"].split()])
                if len(alpha) == 9:
                    polarizabilities.append(np.trace(alpha.reshape(3, 3)) / 3)
                elif len(alpha) == 1:
                    polarizabilities.append(alpha[0])
            except (ValueError, AttributeError):
                pass

        if "gap" in props:
            try:
                gaps.append(float(props["gap"]))
            except (ValueError, AttributeError):
                pass

    print(f"Диполи: {len(dipoles)} значений, "
          f"μ ± σ = {np.mean(dipoles):.3f} ± {np.std(dipoles):.3f} Дебай")
    print(f"Поляризуемости: {len(polarizabilities)} значений, "
          f"μ ± σ = {np.mean(polarizabilities):.3f} ± {np.std(polarizabilities):.3f} Å³")
    print(f"HOMO-LUMO gap: {len(gaps)} значений, "
          f"μ ± σ = {np.mean(gaps):.4f} ± {np.std(gaps):.4f} Eh")

    # Графики
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(dipoles, bins=40, edgecolor="black", color="steelblue")
    axes[0].set_title("Дипольный момент |μ|")
    axes[0].set_xlabel("Дебай")

    axes[1].hist(polarizabilities, bins=40, edgecolor="black", color="darkorange")
    axes[1].set_title("Изотропная поляризуемость tr(α)/3")
    axes[1].set_xlabel("Å³")

    axes[2].hist(gaps, bins=40, edgecolor="black", color="green")
    axes[2].set_title("HOMO-LUMO gap")
    axes[2].set_xlabel("Eh")

    plt.tight_layout()
    plt.savefig("results/figures/eda_targets.png", dpi=100, bbox_inches="tight")
    plt.show()


def demonstrate_symmetry(molecules, n_examples: int = 3):
    """Демонстрация E(3) симметрий: поворот молекулы не меняет свойства."""
    print("\n=== Демонстрация симметрий ===")
    from scipy.spatial.transform import Rotation

    for i in range(min(n_examples, len(molecules))):
        mol = molecules[i]
        coords = np.array([[a[1], a[2], a[3]] for a in mol["atoms"]])

        # Центрируем
        masses = np.array([ATOMIC_MASSES.get(a[0], 1.0) for a in mol["atoms"]])
        com = (coords * masses[:, None]).sum(0) / masses.sum()
        coords = coords - com

        # Поворот на случайный угол
        R = Rotation.random().as_matrix()
        rotated = coords @ R.T

        # Диполь тоже должен повернуться
        if "dipole_moment" in mol["properties"]:
            try:
                mu = np.array([float(v) for v in mol["properties"]["dipole_moment"].split()])
                if len(mu) == 3:
                    mu_rotated = R @ mu
                    print(f"\nМолекула {i}: {len(mol['atoms'])} атомов")
                    print(f"  Диполь исходный: {mu}")
                    print(f"  Диполь повёрнутый: {mu_rotated}")
                    print(f"  |μ| исходный: {np.linalg.norm(mu):.4f}")
                    print(f"  |μ| повёрнутый: {np.linalg.norm(mu_rotated):.4f}")
                    print(f"  (нормы равны → эквивариантность сохранена)")
            except (ValueError, AttributeError):
                pass


if __name__ == "__main__":
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    molecules = load_sample_molecules(n_max=2000)
    analyze_atom_distribution(molecules)
    analyze_targets(molecules)
    demonstrate_symmetry(molecules)
    print("\n=== EDA завершён ===")
    print(f"Графики сохранены в results/figures/")
