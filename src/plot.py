"""Рисование графиков обучения из CSV.

Использование:
  from plot import plot_training_history
  plot_training_history('results/history_egnn_all.csv',
                         save_path='results/figures/egnn_curves.png')

  Или из CLI:
  python src/plot.py --csv results/history_egnn_all.csv --save results/figures/egnn.png
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_history(
    csv_path: str,
    save_path: str | None = None,
    title: str | None = None,
    show: bool = True,
):
    """Построить графики обучения из CSV.

    Args:
        csv_path: путь к results/history_<model>_<target>.csv
        save_path: куда сохранить PNG (None = не сохранять)
        title: заголовок (по умолчанию из имени файла)
        show: показывать ли график (plt.show())
    """
    hist = pd.read_csv(csv_path)

    if title is None:
        # Извлекаем имя модели из имени файла
        fname = os.path.basename(csv_path)
        title = fname.replace("history_", "").replace(".csv", "")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training curves: {title}", fontsize=14, fontweight="bold")

    # 1. Loss
    axes[0, 0].plot(hist["epoch"], hist["train_loss"], label="train", linewidth=2, color="steelblue")
    axes[0, 0].plot(hist["epoch"], hist["val_loss"], label="val", linewidth=2, color="coral")
    axes[0, 0].set_title("Loss (normalized MAE, sum of 3 targets)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. mu MAE
    if "train_mu_mae" in hist.columns:
        axes[0, 1].plot(hist["epoch"], hist["train_mu_mae"], label="train", linewidth=2, color="steelblue")
        axes[0, 1].plot(hist["epoch"], hist["val_mu_mae"], label="val", linewidth=2, color="coral")
    axes[0, 1].set_title("mu MAE (Debye)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. alpha MAE
    if "train_alpha_mae" in hist.columns:
        axes[1, 0].plot(hist["epoch"], hist["train_alpha_mae"], label="train", linewidth=2, color="steelblue")
        axes[1, 0].plot(hist["epoch"], hist["val_alpha_mae"], label="val", linewidth=2, color="coral")
    axes[1, 0].set_title("alpha MAE (a₀³)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. gap MAE
    if "train_gap_mae" in hist.columns:
        axes[1, 1].plot(hist["epoch"], hist["train_gap_mae"], label="train", linewidth=2, color="steelblue")
        axes[1, 1].plot(hist["epoch"], hist["val_gap_mae"], label="val", linewidth=2, color="coral")
    axes[1, 1].set_title("gap MAE (Hartree)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MAE")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохраняем
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def compare_histories(
    csv_paths: list[str],
    labels: list[str] | None = None,
    save_path: str | None = None,
    title: str = "Models comparison",
    show: bool = True,
):
    """Сравнить несколько моделей на одном графике.

    Args:
        csv_paths: list of CSV paths
        labels: подписи для легенды (по умолчанию имена файлов)
        save_path: куда сохранить
    """
    if labels is None:
        labels = [os.path.basename(p).replace("history_", "").replace(".csv", "") for p in csv_paths]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    metrics = [
        ("val_loss", "Loss (normalized)", "val_loss"),
        ("val_mu_mae", "mu MAE (Debye)", "val_mu_mae"),
        ("val_alpha_mae", "alpha MAE (a₀³)", "val_alpha_mae"),
        ("val_gap_mae", "gap MAE (Hartree)", "val_gap_mae"),
    ]

    for ax, (col, title_str, _) in zip(axes.flat, metrics):
        for csv_path, label in zip(csv_paths, labels):
            hist = pd.read_csv(csv_path)
            if col in hist.columns:
                ax.plot(hist["epoch"], hist[col], label=label, linewidth=2)
        ax.set_title(title_str)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"График сравнения сохранён: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="CSV файл истории")
    p.add_argument("--save", type=str, default=None, help="Куда сохранить PNG")
    p.add_argument("--no-show", action="store_true", help="Не показывать (для сервера)")
    args = p.parse_args()

    plot_training_history(args.csv, save_path=args.save, show=not args.no_show)
