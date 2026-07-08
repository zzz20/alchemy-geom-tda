"""Алхимический датасет для PyG.

Читает SDF файлы Alchemy, парсит квантово-механические свойства,
создаёт torch_geometric.data.Data объекты.

Дополнительно вычисляет и кэширует TDA-фичи для каждой молекулы.
"""
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

# Добавляем путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data import parse_alchemy_sdf, mol_to_data, ATOMIC_MASSES, ATOM_TO_IDX, ATOM_TYPES


class AlchemyDataset(InMemoryDataset):
    """PyG датасет для Alchemy.

    Ожидает структуру:
        data/alchemy/
            data/
                *.sdf  (SDF файлы с молекулами)
            properties/
                *.csv (опционально, метаданные)
    """

    ATOM_TYPES = ATOM_TYPES

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        tda_features: bool = False,
        n_bins: int = 16,
        max_radius: float = 5.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.max_samples = max_samples
        self.tda_features = tda_features
        self.n_bins = n_bins
        self.max_radius = max_radius
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # SDF файлы в data/alchemy/data/
        raw_dir = Path(self.root) / "data"
        if not raw_dir.exists():
            return []
        return sorted([f.name for f in raw_dir.glob("*.sdf")])

    @property
    def processed_file_names(self):
        suffix = f"_{self.split}"
        if self.max_samples is not None:
            suffix += f"_max{self.max_samples}"
        if self.tda_features:
            suffix += f"_tda{self.n_bins}"
        return [f"alchemy{suffix}.pt"]

    def download(self):
        # Скачивание делается отдельно через data/download_alchemy.py
        if not (Path(self.root) / "data").exists():
            raise FileNotFoundError(
                f"Датасет Alchemy не найден в {self.root}. "
                "Запустите data/download_alchemy.py"
            )

    def process(self):
        """Парсинг SDF и создание Data объектов."""
        raw_dir = Path(self.root) / "data"
        sdf_files = sorted(raw_dir.glob("*.sdf"))
        if not sdf_files:
            raise FileNotFoundError(f"SDF файлы не найдены в {raw_dir}")

        print(f"[{self.split}] Парсинг {len(sdf_files)} SDF файлов...")

        all_data = []
        for sdf_path in sdf_files:
            molecules = parse_alchemy_sdf(str(sdf_path))
            print(f"  {sdf_path.name}: {len(molecules)} молекул")

            for mol_dict in molecules:
                d = mol_to_data(mol_dict)

                # Создаём Data объект
                data = Data(
                    x=torch.from_numpy(d["x"]),
                    pos=torch.from_numpy(d["pos"]),
                    edge_index=torch.from_numpy(d["edge_index"]),
                    edge_attr=torch.from_numpy(d["edge_attr"]),
                )
                # Таргеты
                if "dipole" in d:
                    data.dipole = torch.from_numpy(d["dipole"])
                if "polarizability" in d:
                    data.polarizability = torch.from_numpy(d["polarizability"])
                if "gap" in d:
                    data.gap = torch.from_numpy(d["gap"])

                # TDA-фичи (если включены)
                if self.tda_features:
                    from src.tda.features import extract_tda_features
                    tda = extract_tda_features(
                        d["pos"], n_bins=self.n_bins, max_radius=self.max_radius
                    )
                    data.tda = torch.from_numpy(tda)

                all_data.append(data)

        # Лимит числа молекул (для отладки)
        if self.max_samples is not None:
            all_data = all_data[:self.max_samples]

        print(f"[{self.split}] Сохраняю {len(all_data)} молекул в {self.processed_paths[0]}")
        self.save(all_data, self.processed_paths[0])


if __name__ == "__main__":
    # Quick test
    ds = AlchemyDataset(root="data/alchemy", split="train", max_samples=10)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Sample 0:")
    print(f"  x: {sample.x.shape}")
    print(f"  pos: {sample.pos.shape}")
    print(f"  edge_index: {sample.edge_index.shape}")
    if hasattr(sample, "dipole"):
        print(f"  dipole: {sample.dipole}")
    if hasattr(sample, "polarizability"):
        print(f"  polarizability: {sample.polarizability.shape}")
