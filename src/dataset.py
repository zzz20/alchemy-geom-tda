"""
PyG датасет для Alchemy.

Читает SDF + final_version.csv, создаёт Data объекты с полями:
  - x, pos, edge_index, edge_attr
  - mu      (1,)  — норма вектора диполя (скаляр)
  - alpha   (1,)  — изотропная поляризуемость (скаляр)
  - gap     (1,)  — HOMO-LUMO gap (скаляр)

Опционально (tda_features=True) добавляет TDA-фичи в поле tda.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import (
    load_properties_csv, find_sdf_files, parse_sdf, mol_to_arrays,
    stratified_split_by_gap,
)


class AlchemyDataset(InMemoryDataset):
    """PyG датасет для Alchemy v20191129.

    Args:
        root: путь к data/alchemy (где лежит папка Alchemy-v20191129)
        split: 'train' | 'val' | 'test' | 'all'
        max_samples: лимит молекул (для отладки)
        tda_features: вычислить и добавить TDA-фичи
        n_bins: число бинов для Betti curves (если tda_features=True)
        max_radius: радиус фильтрации TDA
        seed: сид для split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_samples: int | None = None,
        tda_features: bool = False,
        n_bins: int = 16,
        max_radius: float = 5.0,
        seed: int = 42,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        self.max_samples = max_samples
        self.tda_features = tda_features
        self.n_bins = n_bins
        self.max_radius = max_radius
        self.seed = seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ["Alchemy-v20191129/final_version.csv"]

    @property
    def processed_file_names(self):
        suffix = f"_{self.split}"
        if self.max_samples is not None:
            suffix += f"_max{self.max_samples}"
        if self.tda_features:
            suffix += f"_tda{self.n_bins}"
        return [f"alchemy{suffix}.pt"]

    def download(self):
        if not (Path(self.root) / "Alchemy-v20191129").exists():
            raise FileNotFoundError(
                f"Alchemy не найден в {self.root}. "
                "Запустите: python data/download_alchemy.py"
            )

    def process(self):
        data_root = Path(self.root) / "Alchemy-v20191129"
        csv_path = data_root / "final_version.csv"

        print(f"[{self.split}] Загружаю свойства из {csv_path} ...")
        props = load_properties_csv(csv_path)
        print(f"  Свойств: {len(props)} молекул")

        print(f"[{self.split}] Ищу SDF файлы ...")
        sdf_files = find_sdf_files(data_root)
        print(f"  Найдено SDF: {len(sdf_files)}")

        # Берём только те, у кого есть и SDF, и свойства
        valid_gdb = sorted(set(sdf_files.keys()) & set(props["gdb_idx"].tolist()))
        print(f"  Валидных молекул (SDF + CSV): {len(valid_gdb)}")

        # Лимит для отладки
        if self.max_samples is not None:
            valid_gdb = valid_gdb[:self.max_samples]
            print(f"  Ограничился первыми {self.max_samples}")

        # Разбиение
        if self.split == "all":
            indices = valid_gdb
        else:
            print(f"[{self.split}] Делаю stratified split по gap ...")
            train_idx, val_idx, test_idx = stratified_split_by_gap(
                valid_gdb, props, seed=self.seed
            )
            indices = {"train": train_idx, "val": val_idx, "test": test_idx}[self.split]
            print(f"  {self.split}: {len(indices)} молекул")

        # Строим Data объекты
        data_list = []
        props_dict = props.set_index("gdb_idx").to_dict("index")

        # Опционально TDA
        if self.tda_features:
            from src.tda.features import extract_tda_features

        for i, gdb_idx in enumerate(indices):
            if i % 5000 == 0:
                print(f"  Обработано {i}/{len(indices)}")

            mol = parse_sdf(sdf_files[gdb_idx])
            if mol is None:
                continue

            arr = mol_to_arrays(mol)
            props_row = props_dict[gdb_idx]

            data = Data(
                x=torch.from_numpy(arr["x"]),
                pos=torch.from_numpy(arr["pos"]),
                edge_index=torch.from_numpy(arr["edge_index"]),
                edge_attr=torch.from_numpy(arr["edge_attr"]),
                mu=torch.tensor([props_row["mu"]], dtype=torch.float32),
                alpha=torch.tensor([props_row["alpha"]], dtype=torch.float32),
                gap=torch.tensor([props_row["gap"]], dtype=torch.float32),
                gdb_idx=torch.tensor([gdb_idx], dtype=torch.long),
            )

            if self.tda_features:
                tda = extract_tda_features(
                    arr["pos"], n_bins=self.n_bins, max_radius=self.max_radius
                )
                data.tda = torch.from_numpy(tda).unsqueeze(0)  # (1, 52) — для PyG батчинга

            data_list.append(data)

        print(f"[{self.split}] Сохраняю {len(data_list)} молекул в {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        ds = AlchemyDataset(
            root="data/alchemy",
            split="all",
            max_samples=20,
            tda_features=False,
        )
        print(f"\nDataset size: {len(ds)}")
        sample = ds[0]
        print(f"Sample 0:")
        print(f"  x: {sample.x.shape}")
        print(f"  pos: {sample.pos.shape}")
        print(f"  edge_index: {sample.edge_index.shape}")
        print(f"  mu: {sample.mu}")
        print(f"  alpha: {sample.alpha}")
        print(f"  gap: {sample.gap}")
