"""Основной скрипт обучения.

Примеры запуска:
  # FCNN baseline на диполе
  python src/train.py --model fcnn --target dipole --epochs 50

  # SchNet baseline
  python src/train.py --model schnet --target dipole --epochs 50

  # PaiNN (основная модель)
  python src/train.py --model painn --target dipole --epochs 100

  # PaiNN + TDA (наша финальная модель)
  python src/train.py --model painn_tda --target all --epochs 100

  # Оценка на зашумлённых координатах
  python src/train.py --model painn_tda --target dipole --eval_only \
      --checkpoint checkpoints/painn_tda_best.pt --noise 0.10
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent))

from utils import seed_everything, get_device, AverageMeter, setup_logger
from metrics import (
    dipole_mae, dipole_norm_mae, dipole_angular_error,
    polarizability_iso_mae, polarizability_frobenius_mae,
)
from tda.features import extract_tda_features, tda_feature_dim


def parse_args():
    p = argparse.ArgumentParser(description="Alchemy GeomML + TDA training")
    p.add_argument("--model", type=str, required=True,
                   choices=["fcnn", "schnet", "painn", "painn_tda"],
                   help="Тип модели")
    p.add_argument("--target", type=str, default="dipole",
                   choices=["dipole", "polarizability", "gap", "all"],
                   help="Целевое свойство")
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Шум в координатах при evaluation (для robustness test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true",
                   help="Только оценка (нужен --checkpoint)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_train", type=int, default=None,
                   help="Лимит числа обучающих молекул (для отладки)")
    p.add_argument("--n_bins", type=int, default=16, help="TDA Betti bins")
    return p.parse_args()


def build_model(args, tda_dim: int = 0):
    """Создать модель по аргументам."""
    pred_d = args.target in ("dipole", "all")
    pred_p = args.target in ("polarizability", "all")
    pred_g = args.target in ("gap", "all")

    if args.model == "fcnn":
        from models.fcnn import build_fcnn
        # В FCNN мы предсказываем out_dim компонент
        out_dim = 3 if pred_d else (6 if pred_p else 1)
        return build_fcnn(in_dim=8 * 3, out_dim=out_dim,
                          hidden_dim=args.hidden_channels, n_layers=args.num_layers)

    elif args.model == "schnet":
        from models.schnet import build_schnet
        out_dim = 3 if pred_d else (6 if pred_p else 1)
        return build_schnet(out_dim=out_dim, hidden_channels=args.hidden_channels,
                            num_interactions=args.num_layers, cutoff=args.cutoff)

    elif args.model == "painn":
        from models.painn import build_painn
        return build_painn(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            predict_dipole=pred_d,
            predict_polarizability=pred_p,
            predict_gap=pred_g,
        )

    elif args.model == "painn_tda":
        from models.painn_tda import build_painn_tda
        return build_painn_tda(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            predict_dipole=pred_d,
            predict_polarizability=pred_p,
            predict_gap=pred_g,
            tda_dim=tda_dim or tda_feature_dim(args.n_bins),
        )

    raise ValueError(f"Unknown model: {args.model}")


def compute_loss(preds: dict, batch, target: str) -> torch.Tensor:
    """Вычислить loss для выбранного таргета."""
    loss = 0.0
    if target in ("dipole", "all") and "dipole" in preds:
        loss = loss + dipole_mae(preds["dipole"], batch.dipole)
    if target in ("polarizability", "all") and "polarizability" in preds:
        loss = loss + polarizability_frobenius_mae(preds["polarizability"], batch.polarizability)
    if target in ("gap", "all") and "gap" in preds:
        loss = loss + torch.abs(preds["gap"] - batch.gap).mean()
    return loss


def compute_metrics(preds: dict, batch, target: str) -> dict:
    """Вычислить метрики для логирования."""
    metrics = {}
    if target in ("dipole", "all") and "dipole" in preds:
        metrics["dipole_mae"] = dipole_mae(preds["dipole"], batch.dipole).item()
        metrics["dipole_norm_mae"] = dipole_norm_mae(preds["dipole"], batch.dipole).item()
        metrics["dipole_angle"] = dipole_angular_error(preds["dipole"], batch.dipole).item()
    if target in ("polarizability", "all") and "polarizability" in preds:
        metrics["alpha_iso_mae"] = polarizability_iso_mae(
            preds["polarizability"], batch.polarizability).item()
        metrics["alpha_frob_mae"] = polarizability_frobenius_mae(
            preds["polarizability"], batch.polarizability).item()
    if target in ("gap", "all") and "gap" in preds:
        metrics["gap_mae"] = torch.abs(preds["gap"] - batch.gap).mean().item()
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available()
                          else "cpu" if args.device == "cpu"
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Инициализация логгера
    logger = setup_logger("train", log_file=f"logs/{args.model}_{args.target}.log")

    # Загрузка датасета (заглушка — нужно реализовать AlchemyDataset)
    logger.info("Загрузка датасета Alchemy...")
    from torch_geometric.data import InMemoryDataset, Data
    import os

    # Простой загрузчик — будут реализованы отдельно
    try:
        from data import AlchemyDataset
        train_ds = AlchemyDataset(root=args.data_dir, split="train",
                                  max_samples=args.max_train)
        val_ds = AlchemyDataset(root=args.data_dir, split="val")
        test_ds = AlchemyDataset(root=args.data_dir, split="test")
        logger.info(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    except Exception as e:
        logger.error(f"Не удалось загрузить датасет: {e}")
        logger.error("Сначала запустите data/download_alchemy.py и src/data.py для подготовки")
        return

    # DataLoader (PyG требует DataLoader из torch_geometric.loader)
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # TDA-размерность
    tda_dim = tda_feature_dim(args.n_bins) if args.model == "painn_tda" else 0

    # Создание модели
    model = build_model(args, tda_dim=tda_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Модель: {args.model}, параметров: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Eval-only mode
    if args.eval_only:
        if args.checkpoint is None:
            logger.error("--eval_only требует --checkpoint")
            return
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        test_loop(model, test_loader, device, args, logger, prefix="test")
        return

    # Обучение
    best_val = float("inf")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{args.target}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # === Train ===
        model.train()
        train_loss = AverageMeter()
        for batch in train_loader:
            batch = batch.to(device)
            if args.noise > 0:
                batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
            optimizer.zero_grad()
            preds = model(batch)
            loss = compute_loss(preds, batch, args.target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.update(loss.item(), batch.num_graphs)
        scheduler.step()

        # === Validation ===
        val_metrics = evaluate(model, val_loader, device, args, logger)
        val_loss = val_metrics.get("loss", 0)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss.avg:.4f} | val_loss={val_loss:.4f} | "
            f"{' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items() if k != 'loss')} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  → Сохранён best checkpoint: {ckpt_path}")

    # === Test ===
    logger.info("\n=== Финальная оценка на test ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, args, logger, prefix="test")
    for k, v in test_metrics.items():
        logger.info(f"  test_{k}: {v:.4f}")


def evaluate(model, loader, device, args, logger, prefix="val"):
    """Оценка модели на лоадере."""
    model.eval()
    all_metrics = AverageMeter()
    metric_sums = {}
    counts = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if args.noise > 0 and prefix == "test":
                batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
            preds = model(batch)
            loss = compute_loss(preds, batch, args.target)
            metrics = compute_metrics(preds, batch, args.target)

            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v * batch.num_graphs
            counts += batch.num_graphs
            all_metrics.update(loss.item(), batch.num_graphs)

    avg_metrics = {k: v / max(1, counts) for k, v in metric_sums.items()}
    avg_metrics["loss"] = all_metrics.avg
    return avg_metrics


if __name__ == "__main__":
    main()
