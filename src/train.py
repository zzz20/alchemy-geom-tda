"""Основной скрипт обучения.

Примеры запуска:
  # FCNN baseline
  python src/train.py --model fcnn --target mu --epochs 50

  # SchNet baseline
  python src/train.py --model schnet --target mu --epochs 50

  # PaiNN (основная модель)
  python src/train.py --model painn --target all --epochs 100

  # PaiNN + TDA (наша финальная модель)
  python src/train.py --model painn_tda --target all --epochs 100

  # Оценка на зашумлённых координатах
  python src/train.py --model painn_tda --target all --eval_only \
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

sys.path.insert(0, str(Path(__file__).parent))

from utils import seed_everything, get_device, AverageMeter, setup_logger
from metrics import mae, mu_mae, alpha_mae, gap_mae
from tda.features import extract_tda_features, tda_feature_dim


def parse_args():
    p = argparse.ArgumentParser(description="Alchemy GeomML + TDA training")
    p.add_argument("--model", type=str, required=True,
                   choices=["fcnn", "schnet", "painn", "painn_tda", "egnn", "egnn_tda", "egnn_vector"],
                   help="Тип модели")
    p.add_argument("--target", type=str, default="all",
                   choices=["mu", "alpha", "gap", "all"],
                   help="Целевое свойство")
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)  # НИЖЕ! 5e-4 не учится
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Шум в координатах (для robustness test)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true",
                   help="Только оценка (нужен --checkpoint)")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_train", type=int, default=None,
                   help="Лимит числа обучающих молекул (для отладки)")
    p.add_argument("--max_val", type=int, default=None,
                   help="Лимит валидационных молекул (для отладки)")
    p.add_argument("--max_test", type=int, default=None,
                   help="Лимит тестовых молекул (для отладки)")
    p.add_argument("--n_bins", type=int, default=16, help="TDA Betti bins")
    p.add_argument("--max_radius", type=float, default=5.0, help="TDA радиус")
    return p.parse_args()


def build_model(args, tda_dim: int = 0):
    """Создать модель по аргументам."""
    pred_mu = args.target in ("mu", "all")
    pred_alpha = args.target in ("alpha", "all")
    pred_gap = args.target in ("gap", "all")

    if args.model == "fcnn":
        from models.fcnn import build_fcnn
        out_dim = 3 if args.target == "all" else 1
        return build_fcnn(in_dim=8 * 3, out_dim=out_dim,
                          hidden_dim=args.hidden_channels, n_layers=args.num_layers)

    elif args.model == "schnet":
        from models.schnet import build_schnet
        out_dim = 3 if args.target == "all" else 1
        return build_schnet(out_dim=out_dim, hidden_channels=args.hidden_channels,
                            num_interactions=args.num_layers, cutoff=args.cutoff)

    elif args.model == "painn":
        from models.painn import build_painn
        return build_painn(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "painn_tda":
        from models.painn_tda import build_painn_tda
        return build_painn_tda(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
            tda_dim=tda_dim or tda_feature_dim(args.n_bins),
        )

    elif args.model == "egnn":
        from models.egnn import build_egnn
        return build_egnn(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_tda":
        from models.egnn_tda import build_egnn_tda
        return build_egnn_tda(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            tda_dim=tda_dim or tda_feature_dim(args.n_bins),
            predict_mu=pred_mu,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    elif args.model == "egnn_vector":
        from models.egnn_vector import build_egnn_vector
        return build_egnn_vector(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
            predict_alpha=pred_alpha,
            predict_gap=pred_gap,
        )

    raise ValueError(f"Unknown model: {args.model}")


def _unpack_preds(preds, target: str) -> dict:
    """Унификация: FCNN/SchNet возвращают тензор, PaiNN — словарь."""
    if isinstance(preds, dict):
        return preds
    if target == "all":
        return {"mu": preds[:, 0:1], "alpha": preds[:, 1:2], "gap": preds[:, 2:3]}
    return {target: preds}


def _get_target(key: str, batch, target_stats: dict | None = None):
    """Получить таргет по ключу. Для векторного mu — возвращаем как есть (B, 3).
    Для скаляров — нормализуем если есть target_stats."""
    val = getattr(batch, key)
    if target_stats is not None and key in target_stats:
        m, s = target_stats[key]
        return (val - m) / s
    return val


def compute_loss(preds, batch, target: str, target_stats: dict | None = None) -> torch.Tensor:
    """Вычислить loss. Для векторного mu (B,3) — MAE по каждой компоненте."""
    preds = _unpack_preds(preds, target)

    loss = 0.0
    for key in ["mu", "alpha", "gap"]:
        if target not in (key, "all"):
            continue
        if key not in preds:
            continue
        pred_val = preds[key]
        target_val = _get_target(key, batch, target_stats)
        # Если pred векторный (B,3) а target скалярный (B,1) — берём норму предсказания
        if pred_val.dim() == 2 and pred_val.shape[1] == 3 and target_val.dim() == 2 and target_val.shape[1] == 1:
            # Векторный mu: сравниваем норму вектора со скалярным таргетом
            pred_norm = pred_val.norm(dim=-1, keepdim=True)
            loss = loss + (pred_norm - target_val).abs().mean()
        else:
            loss = loss + (pred_val - target_val).abs().mean()
    return loss


def compute_metrics(preds, batch, target: str, target_stats: dict | None = None) -> dict:
    """Вычислить метрики в исходных единицах."""
    preds = _unpack_preds(preds, target)
    metrics = {}

    for key in ["mu", "alpha", "gap"]:
        if target not in (key, "all"):
            continue
        if key not in preds:
            continue
        pred_val = preds[key]
        target_val = getattr(batch, key)

        # Денормализуем предсказание
        if target_stats is not None and key in target_stats:
            mean, std = target_stats[key]
            if pred_val.dim() == 2 and pred_val.shape[1] == 3:
                # Векторный mu — денормализуем норму
                pred_norm = pred_val.norm(dim=-1, keepdim=True)
                pred_val = pred_norm * std + mean
            else:
                pred_val = pred_val * std + mean

        # Сравнение
        if pred_val.dim() == 2 and pred_val.shape[1] == 1:
            metrics[f"{key}_mae"] = (pred_val - target_val).abs().mean().item()
        elif pred_val.dim() == 2 and pred_val.shape[1] == 3:
            # Векторный mu — сравниваем норму со скалярным таргетом
            pred_norm = pred_val.norm(dim=-1, keepdim=True)
            metrics[f"{key}_mae"] = (pred_norm - target_val).abs().mean().item()
        else:
            metrics[f"{key}_mae"] = (pred_val - target_val).abs().mean().item()
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    logger = setup_logger("train", log_file=f"logs/{args.model}_{args.target}.log")
    # Загрузка датасета
    logger.info("Загрузка датасета Alchemy...")
    from dataset import AlchemyDataset

    use_tda = args.model in ("painn_tda", "egnn_tda")
    train_ds = AlchemyDataset(root=args.data_dir, split="train",
                              max_samples=args.max_train,
                              tda_features=use_tda, n_bins=args.n_bins,
                              max_radius=args.max_radius, seed=args.seed)
    val_ds = AlchemyDataset(root=args.data_dir, split="val",
                            max_samples=args.max_val,
                            tda_features=use_tda, n_bins=args.n_bins,
                            max_radius=args.max_radius, seed=args.seed)
    test_ds = AlchemyDataset(root=args.data_dir, split="test",
                             max_samples=args.max_test,
                             tda_features=use_tda, n_bins=args.n_bins,
                             max_radius=args.max_radius, seed=args.seed)
    logger.info(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # === Нормализация таргетов (по train выборке) ===
    # Считаем mean/std для mu, alpha, gap и храним в словаре
    target_stats = {}
    for key in ["mu", "alpha", "gap"]:
        vals = torch.cat([getattr(d, key) for d in train_ds])
        target_stats[key] = (float(vals.mean()), float(vals.std() + 1e-8))
    logger.info(f"Target stats (mean, std): {target_stats}")

    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    tda_dim = tda_feature_dim(args.n_bins) if args.model in ("painn_tda", "egnn_tda") else 0

    model = build_model(args, tda_dim=tda_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Модель: {args.model}, параметров: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.eval_only:
        if args.checkpoint is None:
            logger.error("--eval_only требует --checkpoint")
            return
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        test_metrics = evaluate(model, test_loader, device, args, logger, prefix="test")
        for k, v in test_metrics.items():
            logger.info(f"  test_{k}: {v:.4f}")
        return

    best_val = float("inf")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_{args.target}_best.pt"

    # === История обучения для графиков ===
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # === Train ===
        model.train()
        train_loss = AverageMeter()
        # Дополнительно: train метрики
        train_metric_sums = {}
        train_counts = 0
        for batch in train_loader:
            batch = batch.to(device)
            if args.noise > 0:
                batch.pos = batch.pos + torch.randn_like(batch.pos) * args.noise
            optimizer.zero_grad()
            preds = model(batch)
            loss = compute_loss(preds, batch, args.target, target_stats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss.update(loss.item(), batch.num_graphs)

            # Train метрики (без backward, для отслеживания)
            with torch.no_grad():
                tr_metrics = compute_metrics(preds, batch, args.target, target_stats)
                for k, v in tr_metrics.items():
                    train_metric_sums[k] = train_metric_sums.get(k, 0.0) + v * batch.num_graphs
                train_counts += batch.num_graphs
        scheduler.step()

        train_avg_metrics = {k: v / max(1, train_counts) for k, v in train_metric_sums.items()}

        # === Validation ===
        val_metrics = evaluate(model, val_loader, device, args, logger, target_stats=target_stats)
        val_loss = val_metrics.get("loss", 0)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss.avg:.4f} | val_loss={val_loss:.4f} | "
            f"{' | '.join(f'{k}={v:.4f}' for k, v in val_metrics.items() if k != 'loss')} | "
            f"{elapsed:.1f}s"
        )

        # Сохраняем историю
        row = {
            "epoch": epoch,
            "train_loss": train_loss.avg,
            "val_loss": val_loss,
            "elapsed": elapsed,
        }
        for k, v in train_avg_metrics.items():
            row[f"train_{k}"] = v
        for k, v in val_metrics.items():
            if k != "loss":
                row[f"val_{k}"] = v
        history.append(row)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  → Сохранён best checkpoint: {ckpt_path}")

    # === Test ===
    logger.info("\n=== Финальная оценка на test ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, args, logger, prefix="test", target_stats=target_stats)
    for k, v in test_metrics.items():
        logger.info(f"  test_{k}: {v:.4f}")

    # === Сохранение истории в CSV ===
    import csv
    csv_path = f"results/history_{args.model}_{args.target}.csv"
    if history:
        keys = list(history[0].keys())
        # Добавляем test-метрики в последнюю строку
        for k, v in test_metrics.items():
            history[-1][f"test_{k}"] = v
        keys.extend([f"test_{k}" for k in test_metrics.keys()])
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        logger.info(f"История сохранена в {csv_path}")


def evaluate(model, loader, device, args, logger, prefix="val", target_stats: dict | None = None):
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
            loss = compute_loss(preds, batch, args.target, target_stats)
            metrics = compute_metrics(preds, batch, args.target, target_stats)

            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v * batch.num_graphs
            counts += batch.num_graphs
            all_metrics.update(loss.item(), batch.num_graphs)

    avg_metrics = {k: v / max(1, counts) for k, v in metric_sums.items()}
    avg_metrics["loss"] = all_metrics.avg
    return avg_metrics


if __name__ == "__main__":
    main()
