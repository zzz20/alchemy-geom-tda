"""Единый скрипт запуска всех моделей.

Запуск:
  python src/run_all.py --epochs 100 --max_train 10000 --max_val 1000 --max_test 1000

Для теста (быстро):
  python src/run_all.py --epochs 5 --max_train 200 --max_val 50 --max_test 50
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    p = argparse.ArgumentParser(description="Run all models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max_train", type=int, default=10000)
    p.add_argument("--max_val", type=int, default=1000)
    p.add_argument("--max_test", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--data_dir", type=str, default="data/alchemy")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default="all",
                   help="Список моделей через запятую или 'all'")
    args = p.parse_args()

    # Список моделей
    if args.models == "all":
        models = ["fcnn", "schnet", "egnn", "egnn_tda", "egnn_vector", "egnn_vector_tda"]
    else:
        models = args.models.split(",")

    results = {}

    for model_name in models:
        print(f"\n{'='*70}")
        print(f"  ЗАПУСК: {model_name}")
        print(f"{'='*70}\n")

        # Очищаем кэш для чистоты (кроме TDA — он долго вычисляется)
        # Кэш автоматически переиспользуется если параметры те же

        # Аргументы для train.py
        argv = [
            'train.py',
            '--model', model_name,
            '--target', 'all',
            '--epochs', str(args.epochs),
            '--max_train', str(args.max_train),
            '--max_val', str(args.max_val),
            '--max_test', str(args.max_test),
            '--batch_size', str(args.batch_size),
            '--hidden_channels', str(args.hidden_channels),
            '--num_layers', str(args.num_layers),
            '--device', args.device,
            '--lr', str(args.lr),
            '--seed', str(args.seed),
            '--data_dir', args.data_dir,
            '--patience', str(args.patience),
            '--lr_patience', str(args.lr_patience),
        ]

        # TDA-специфичные параметры
        if model_name in ("egnn_tda", "painn_tda", "egnn_vector_tda"):
            argv.extend(['--n_bins', '16'])

        sys.argv = argv

        # Импортируем и запускаем
        import importlib
        if 'train' in sys.modules:
            importlib.reload(sys.modules['train'])

        t0 = time.time()
        try:
            from train import main as train_main
            train_main()
            elapsed = time.time() - t0
            results[model_name] = {"status": "OK", "time": elapsed}
            print(f"\n✓ {model_name} завершён за {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[model_name] = {"status": f"ERROR: {e}", "time": elapsed}
            print(f"\n✗ {model_name} ОШИБКА: {e}")

    # Сводка
    print(f"\n{'='*70}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*70}")
    for name, res in results.items():
        print(f"  {name:15s}: {res['status']:20s} ({res['time']:.1f}s)")


if __name__ == "__main__":
    main()
