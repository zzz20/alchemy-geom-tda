"""Единый скрипт запуска всех моделей.

v27 изменения:
  - batch_size по умолчанию 1024 (раньше 256)
  - --output_dir: куда складывать CSV (по умолчанию results/experiments/batch_size_{bs})
  - --input_dir: алиас для --data_dir (для совместимости)
  - передаёт --output_dir в train.py

Запуск:
  python src/run_all.py --epochs 100 --batch_size 1024

  С явным указанием output_dir:
  python src/run_all.py --batch_size 1024 --output_dir results/experiments/batch_size_1024

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
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_val", type=int, default=None)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1024)  # v27: было 256
    p.add_argument("--hidden_channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr_patience", type=int, default=5)
    p.add_argument("--data_dir", type=str, default="data/alchemy",
                   help="Путь к датасету (input_dir алиас)")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Алиас для --data_dir (v27)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Куда складывать CSV-истории. "
                        "По умолчанию results/experiments/batch_size_{bs}")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default="all",
                   help="Список моделей через запятую или 'all'")
    args = p.parse_args()

    # v27: input_dir — алиас для data_dir
    if args.input_dir is not None:
        args.data_dir = args.input_dir

    # v27: auto output_dir по batch_size
    if args.output_dir is None:
        args.output_dir = f"results/experiments/batch_size_{args.batch_size}"

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

        # Аргументы для train.py
        argv = [
            'train.py',
            '--model', model_name,
            '--target', 'all',
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--hidden_channels', str(args.hidden_channels),
            '--num_layers', str(args.num_layers),
            '--device', args.device,
            '--lr', str(args.lr),
            '--seed', str(args.seed),
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,  # v27
            '--patience', str(args.patience),
            '--lr_patience', str(args.lr_patience),
        ]

        # Передаём max_ параметры только если они заданы (не None)
        if args.max_train is not None:
            argv.extend(['--max_train', str(args.max_train)])
        if args.max_val is not None:
            argv.extend(['--max_val', str(args.max_val)])
        if args.max_test is not None:
            argv.extend(['--max_test', str(args.max_test)])

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
            print(f"\n[OK] {model_name} завершён за {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[model_name] = {"status": f"ERROR: {e}", "time": elapsed}
            print(f"\n[FAIL] {model_name} ОШИБКА: {e}")
            import traceback
            traceback.print_exc()

    # v28: собираем summary CSV из последних history каждого model run
    summary_rows = []
    import glob, os, pandas as pd
    for model_name in models:
        # Ищем все history-CSV этой модели, берём самый свежий по mtime
        csvs = sorted(glob.glob(f"{args.output_dir}/history_{model_name}_*.csv"),
                      key=os.path.getmtime)
        if not csvs:
            summary_rows.append({
                'model': model_name,
                'status': results.get(model_name, {}).get('status', 'NO_CSV'),
                'train_time_s': results.get(model_name, {}).get('time', 0.0),
                'n_epochs': 0,
                'best_epoch': None,
                'best_val_loss': None,
                'test_mu_mae': None,
                'test_alpha_mae': None,
                'test_gap_mae': None,
                'test_loss': None,
                'csv_file': None,
            })
            continue
        latest = csvs[-1]
        try:
            df = pd.read_csv(latest)
            last_row = df.iloc[-1]
            best_idx = df['val_loss'].idxmin() if 'val_loss' in df.columns else 0
            summary_rows.append({
                'model': model_name,
                'status': results.get(model_name, {}).get('status', 'OK'),
                'train_time_s': round(results.get(model_name, {}).get('time', 0.0), 1),
                'n_epochs': len(df),
                'best_epoch': int(df.loc[best_idx, 'epoch']) if 'epoch' in df.columns else None,
                'best_val_loss': round(float(df.loc[best_idx, 'val_loss']), 4) if 'val_loss' in df.columns else None,
                'test_mu_mae': round(float(last_row.get('test_mu_mae', float('nan'))), 4) if 'test_mu_mae' in df.columns else None,
                'test_alpha_mae': round(float(last_row.get('test_alpha_mae', float('nan'))), 4) if 'test_alpha_mae' in df.columns else None,
                'test_gap_mae': round(float(last_row.get('test_gap_mae', float('nan'))), 4) if 'test_gap_mae' in df.columns else None,
                'test_loss': round(float(last_row.get('test_loss', float('nan'))), 4) if 'test_loss' in df.columns else None,
                'csv_file': os.path.basename(latest),
            })
        except Exception as e:
            print(f"[WARN] Не удалось прочитать {latest}: {e}")

    # Сохраняем summary CSV
    from datetime import datetime
    summary_df = pd.DataFrame(summary_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{args.output_dir}/summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)

    # Сводка в stdout
    print(f"\n{'='*70}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*70}")
    for name, res in results.items():
        print(f"  {name:20s}: {res['status']:30s} ({res['time']:.1f}s)")

    print(f"\n=== Summary CSV сохранён: {summary_path} ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
