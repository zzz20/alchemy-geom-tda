"""
Загрузка датасета Alchemy.

Источник: https://github.com/tencent-alchemy/Alchemy

Скачивает:
- data/alchemy/supplementary/  (SD files с молекулами)
- data/alchemy/labels/         (CSV с квантово-механическими свойствами)

Доступные свойства (12 шт.):
- dipole_moment (μ, вектор 1x3, Дебай)        ← наш главный таргет
- polarizability (α, тензор 3x3, Å³)          ← наш второй таргет
- homo, lumo, gap                              ← скаляры (доп. multi-task)
- R2, zpve, U0, U, H, G, Cv                    ← скаляры
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# URL репозитория Tencent Alchemy
# Скачиваем zip-архив ветки master
ALCHEMY_REPO = "https://github.com/tencent-alchemy/Alchemy/archive/refs/heads/master.zip"

DATA_DIR = Path(__file__).parent / "alchemy"


def download_alchemy(force: bool = False) -> None:
    """Скачать и распаковать датасет Alchemy.

    Args:
        force: перекачать даже если данные уже есть
    """
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()) and not force:
        print(f"[OK] Alchemy уже скачан в {DATA_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "alchemy_repo.zip"

    print(f"[1/3] Скачиваю Alchemy из {ALCHEMY_REPO} ...")
    print(f"      Размер ~ 200 MB, может занять несколько минут.")
    urllib.request.urlretrieve(ALCHEMY_REPO, zip_path)
    print(f"      Сохранено в {zip_path}")

    print(f"[2/3] Распаковываю ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # Перемещаем содержимое Alchemy-master/* в data/alchemy/
    extracted = DATA_DIR / "Alchemy-master"
    if extracted.exists():
        for item in extracted.iterdir():
            target = DATA_DIR / item.name
            if target.exists():
                continue
            item.rename(target)
        extracted.rmdir()

    zip_path.unlink()
    print(f"[3/3] Готово. Данные в {DATA_DIR}")

    # Показать содержимое
    print("\nСодержимое data/alchemy/:")
    for item in DATA_DIR.iterdir():
        print(f"  {item.name}/ " if item.is_dir() else f"  {item.name}")


def inspect_dataset() -> None:
    """Краткий осмотр структуры датасета."""
    print("\n=== Структура Alchemy ===\n")
    for sub in ["data", "graphs", "properties"]:
        sub_path = DATA_DIR / sub
        if sub_path.exists():
            files = list(sub_path.iterdir())[:5]
            print(f"data/alchemy/{sub}/")
            for f in files:
                size = f.stat().st_size / 1e6 if f.is_file() else 0
                print(f"  {f.name}" + (f" ({size:.1f} MB)" if size else ""))
            n_total = len(list(sub_path.iterdir()))
            if n_total > 5:
                print(f"  ... и ещё {n_total - 5} файлов")
            print()


if __name__ == "__main__":
    download_alchemy(force="--force" in sys.argv)
    inspect_dataset()
