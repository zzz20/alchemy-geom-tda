# Alchemy GeomML + TDA

Предсказание **вектора дипольного момента** μ (1×3) и **тензора поляризуемости** α (3×3) молекул датасета [Alchemy](https://arxiv.org/pdf/1906.09427) с использованием:

- **Геометрического ML:** E(3)-эквивариантная нейросеть PaiNN (с векторными и тензорными выходами)
- **Топологического анализа данных (TDA):** персистентная гомология облака 3D-точек атомов (Vietoris-Rips) + интеграция через FiLM conditioning
- **Baselines:** FCNN, SchNet

## Симметрии задачи

| Свойство | Тип | Представление | Преобразование при вращении R |
|---|---|---|---|
| Диполь μ | вектор 1×3 | irrep l=1 | μ → R·μ |
| Изотропная поляризуемость tr(α) | скаляр | irrep l=0 | не меняется |
| Анизотропная поляризуемость α_aniso | тензор 5 компонент | irrep l=2 | α → R·α·Rᵀ |
| HOMO-LUMO gap | скаляр | irrep l=0 | не меняется |

Сеть строго эквивариантна к E(3) = (сдвиги) ⋊ O(3).

## Архитектура

```
                   ┌─────────────────────┐
                   │  TDA-модуль         │
   3D координаты ──┤  Vietoris-Rips      │── TDA-фичи (130D) ──┐
   атомов          │  Betti curves       │                      │ FiLM
                   │  Persistence entropy│                      │ conditioning
                   └─────────────────────┘                      │
                                                                ▼
   Атомы +         ┌─────────────────────┐                ┌──────────────┐
   координаты  ───▶│  PaiNN              │───────────────▶│  Heads       │──▶ μ, α, gap
                   │  (E(3)-эквивариант) │                │  (l=0,1,2)   │
                   └─────────────────────┘                └──────────────┘
```

## Установка

```bash
# Создаём окружение
conda create -n alchemy python=3.10 -y
conda activate alchemy

# PyTorch (CPU или GPU — заменить на нужную версию)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric

# TDA и химия
pip install gudhi rdkit

# Прочее
pip install numpy pandas scipy matplotlib seaborn tqdm pyyaml scikit-learn
```

Или:
```bash
pip install -r requirements.txt
```

## Использование

### 1. Загрузка датасета Alchemy

```bash
python data/download_alchemy.py
```

### 2. Обучение моделей

```bash
# FCNN baseline
python src/train.py --model fcnn --target dipole

# SchNet baseline
python src/train.py --model schnet --target dipole

# PaiNN (основная модель)
python src/train.py --model painn --target dipole

# PaiNN + TDA
python src/train.py --model painn_tda --target dipole

# Multi-task: диполь + поляризуемость + HOMO-LUMO
python src/train.py --model painn_tda --target all
```

### 3. Тестирование робастности к шуму

```bash
python src/train.py --model painn_tda --target dipole --noise 0.10 --eval_only --checkpoint best.pt
```

## Структура репозитория

```
alchemy-geom-tda/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── download_alchemy.py
├── src/
│   ├── data.py                # Alchemy dataset, сплиты, нормализация
│   ├── utils.py               # seeds, logging
│   ├── metrics.py             # MAE, угловая ошибка
│   ├── train.py               # основной скрипт обучения
│   ├── models/
│   │   ├── fcnn.py            # FCNN baseline
│   │   ├── schnet.py          # SchNet baseline
│   │   ├── painn.py           # PaiNN с l=0,1,2 выходами
│   │   └── painn_tda.py       # PaiNN + TDA через FiLM
│   └── tda/
│       ├── features.py        # Vietoris-Rips, Betti curves
│       └── film.py            # FiLM conditioning
├── configs/
│   └── default.yaml
└── notebooks/
    └── 01_eda.py
```

## Результаты

См. `results/table.md` после обучения.

## Ссылки

- **Alchemy dataset:** Chen et al., 2019. [arXiv:1906.09427](https://arxiv.org/pdf/1906.09427)
- **PaiNN:** Schütt et al., 2021. [arXiv:2102.03150](https://arxiv.org/abs/2102.03150)
- **Equivariant ML обзор:** Weiler, 2023. [блог](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- **TDA обзор:** Chazal & Michel, 2019. [arXiv:1904.11044](https://arxiv.org/pdf/1904.11044)
