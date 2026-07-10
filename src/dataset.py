"""
PyG датасет для Alchemy.

ИСПРАВЛЕНО v24:
  1. Утечка данных — train/val/test теперь используют РАЗНЫЕ молекулы
  2. Размеры val/test — max_val=1000 даёт 1000, а не 100
  3. Логика: сначала split всего датасета, потом ограничение размеров
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
        max_samples: лимит молекул В ЭТОМ СПЛИТЕ (а не в общем пуле)
        tda_features: вычислить и добавить TDA-фичи
        n_bins: число бинов для Betti curves
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
        return [f"alchemy_v24{suffix}.pt"]

    def download(self):
        if not (Path(self.root) / "Alchemy-v20191129").exists():
            raise FileNotFoundError(
                f"Alchemy не найден в {self.root}. "
                "Запустите: python data/download_alchemy.py"
            )

    def process(self):
        """Парсинг SDF и создание Data объектов.

        ИСПРАВЛЕНО: split делается ОДИН РАЗ для всего датасета,
        потом max_samples ограничивает каждый сплит отдельно.
        Никакой утечки данных.
        """
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

        # === ИСПРАВЛЕНИЕ: делаем split ОДИН РАЗ для ВСЕХ молекул ===
        print(f"[{self.split}] Делаю stratified split по gap ...")
        train_idx, val_idx, test_idx = stratified_split_by_gap(
            valid_gdb, props, seed=self.seed
        )
        print(f"  Всего: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # Выбираем нужный сплит
        if self.split == "all":
            indices = valid_gdb
        elif self.split == "train":
            indices = train_idx
        elif self.split == "val":
            indices = val_idx
        elif self.split == "test":
            indices = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # === ИСПРАВЛЕНИЕ: max_samples ограничивает УЖЕ выбранный сплит ===
        if self.max_samples is not None:
            indices = indices[:self.max_samples]
        print(f"  {self.split}: {len(indices)} молекул (после max_samples)")

        # Строим Data объекты
        data_list = []
        props_dict = props.set_index("gdb_idx").to_dict("index")

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
                data.tda = torch.from_numpy(tda).unsqueeze(0)  # (1, 52) для PyG

            data_list.append(data)

        print(f"[{self.split}] Сохраняю {len(data_list)} молекул в {self.processed_paths[0]}")
        self.save(data_list, self.processed_paths[0])
