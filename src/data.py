"""
Загрузка и препроцессинг датасета Alchemy.

Alchemy содержит 119,487 молекул в SD file формате с 12 квантово-механическими
свойствами. Мы предсказываем:
  - dipole_moment μ (вектор 1×3, Дебай)        ← l=1
  - polarizability α (тензор 3×3, Å³)          ← l=0 ⊕ l=2
  - homo_lumo_gap (скаляр, Eh)                 ← l=0 (опционально, multi-task)

Структура Data объекта PyG для одной молекулы:
  - x:           (N, F)  узловые признаки [atom_type one-hot, atomic_num, ...]
  - pos:         (N, 3)  3D координаты атомов
  - edge_index:  (2, E)  связи (из SDF)
  - edge_attr:  (E, F_e) признаки связей (тип one-hot)
  - y:           (T,)    таргеты
  - dipole:      (3,)    вектор диполя (для удобства отдельным полем)
  - polarizability: (3,3) тензор поляризуемости
  - gap:         (1,)    HOMO-LUMO gap
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# Химические элементы в Alchemy: H, C, N, O, F, S, Cl
ATOM_TYPES = ["H", "C", "N", "O", "F", "S", "Cl"]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

# Массы атомов (для центра масс и перевода в центр)
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "S": 32.06, "Cl": 35.45,
}

# Максимальное число атомов в Alchemy (14 heavy + H ≈ 30)
MAX_ATOMS = 30


def parse_alchemy_sdf(sdf_path: str) -> list[dict]:
    """Парсит SDF файл Alchemy в список молекул.

    Возвращает список словарей:
      {
        'conformer': RDKit conformer,
        'atoms': list of (symbol, x, y, z),
        'bonds': list of (i, j, bond_type),
        'properties': dict of property_name -> value
      }
    """
    from rdkit import Chem

    molecules = []
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    for mol in suppl:
        if mol is None:
            continue

        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()

        atoms = []
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

        bonds = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            btype = bond.GetBondTypeAsDouble()
            bonds.append((i, j, btype))

        # Свойства из SDF
        props = {}
        for name in mol.GetPropsAsDict():
            props[name] = mol.GetProp(name)

        molecules.append({
            "rdkit_mol": mol,
            "atoms": atoms,
            "bonds": bonds,
            "properties": props,
        })

    return molecules


def atoms_to_features(symbol: str) -> np.ndarray:
    """Узловой признак: one-hot по типу атома + atomic number.

    Размерность = len(ATOM_TYPES) + 1 = 8.
    """
    feat = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
    if symbol in ATOM_TO_IDX:
        feat[ATOM_TO_IDX[symbol]] = 1.0
        feat[-1] = ATOMIC_MASSES.get(symbol, 0.0)
    return feat


def bonds_to_edge_index(bonds: list, n_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Преобразовать список связей в edge_index и edge_attr.

    Возвращает:
      edge_index: (2, 2*E) — оба направления
      edge_attr:  (2*E, 4) — one-hot по типу связи [single, double, triple, aromatic]
    """
    if not bonds:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    src, dst, types = [], [], []
    for i, j, btype in bonds:
        src.extend([i, j])
        dst.extend([j, i])
        types.extend([btype, btype])

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.zeros((len(src), 4), dtype=np.float32)
    for k, btype in enumerate(types):
        # 1.0=single, 1.5=aromatic, 2.0=double, 3.0=triple
        if btype == 1.0:
            edge_attr[k, 0] = 1.0
        elif btype == 1.5:
            edge_attr[k, 3] = 1.0
        elif btype == 2.0:
            edge_attr[k, 1] = 1.0
        elif btype == 3.0:
            edge_attr[k, 2] = 1.0
    return edge_index, edge_attr


def mol_to_data(mol_dict: dict) -> dict:
    """Преобразовать молекулу в словарь тензоров (без torch_geometric.Data).

    Возвращает словарь с numpy массивами:
      x, pos, edge_index, edge_attr,
      dipole, polarizability, gap
    """
    atoms = mol_dict["atoms"]
    n = len(atoms)

    # Узловые признаки
    x = np.stack([atoms_to_features(a[0]) for a in atoms])  # (N, 8)

    # Координаты
    pos = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=np.float32)  # (N, 3)

    # Центрируем в центр масс (трансляционная инвариантность)
    masses = np.array([ATOMIC_MASSES.get(a[0], 1.0) for a in atoms], dtype=np.float32)
    com = (pos * masses[:, None]).sum(0) / masses.sum()
    pos = pos - com

    # Связи
    edge_index, edge_attr = bonds_to_edge_index(mol_dict["bonds"], n)

    # Таргеты — извлекаем из свойств
    props = mol_dict["properties"]
    out = {"x": x, "pos": pos, "edge_index": edge_index, "edge_attr": edge_attr}

    # Dipole moment — в Alchemy хранится как 3 числа "dx dy dz"
    # В SD файлах Alchemy это поле 'dipole_moment'
    if "dipole_moment" in props:
        try:
            mu = np.array([float(v) for v in props["dipole_moment"].split()], dtype=np.float32)
            if len(mu) == 3:
                out["dipole"] = mu
            elif len(mu) == 1:
                out["dipole_norm"] = mu[0]
        except (ValueError, AttributeError):
            pass

    # Polarizability — тензор 3×3, хранится как 9 чисел
    if "polarizability" in props:
        try:
            alpha = np.array([float(v) for v in props["polarizability"].split()], dtype=np.float32)
            if len(alpha) == 9:
                out["polarizability"] = alpha.reshape(3, 3)
            elif len(alpha) == 1:
                out["polarizability_iso"] = alpha[0]
        except (ValueError, AttributeError):
            pass

    # HOMO-LUMO gap
    if "gap" in props:
        try:
            out["gap"] = np.array([float(props["gap"])], dtype=np.float32)
        except (ValueError, AttributeError):
            pass

    return out


def normalize_targets(samples: list[dict], keys: list[str]) -> dict[str, dict]:
    """Вычислить статистики (mean, std) для нормализации таргетов на train выборке.

    Возвращает словарь key -> {"mean": ..., "std": ...}
    """
    stats = {}
    for key in keys:
        values = []
        for s in samples:
            if key in s:
                v = s[key]
                if v.ndim == 0:
                    values.append(v)
                elif v.ndim == 1:
                    values.append(v)
                elif v.ndim == 2:
                    # тензор: flatten
                    values.append(v.flatten())
        if not values:
            continue
        all_v = np.concatenate(values)
        stats[key] = {
            "mean": float(all_v.mean()),
            "std": float(all_v.std() + 1e-8),
        }
    return stats


def stratified_split(
    samples: list[dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[list[int], list[int], list[int]]:
    """Stratified split по HOMO-LUMO gap (как в статье Alchemy).

    Сортируем по gap, разбиваем на группы по 10, в каждой 8/1/1.
    """
    rng = np.random.default_rng(seed)
    n = len(samples)

    # Сортируем индексы по gap
    def get_gap(i):
        return float(samples[i].get("gap", [0.0])[0])

    idx_sorted = sorted(range(n), key=get_gap)

    train, val, test = [], [], []
    chunk = 10
    for i in range(0, n, chunk):
        block = idx_sorted[i:i + chunk]
        rng.shuffle(block)
        n_train = int(len(block) * 0.8)
        n_val = int(len(block) * 0.1)
        train.extend(block[:n_train])
        val.extend(block[n_train:n_train + n_val])
        test.extend(block[n_train + n_val:])

    return train, val, test


if __name__ == "__main__":
    # Quick test
    print("Этот модуль содержит функции для загрузки Alchemy.")
    print("Запустите data/download_alchemy.py сначала, чтобы скачать данные.")
