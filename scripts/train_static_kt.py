from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from static_kt.common import (
    compute_metrics,
    discover_datasets,
    get_sequences,
    predict_dataframe,
    set_seed,
    split_train_val_ensure_all_items,
    train_static_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StaticKT and save test predictions.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/static_kt"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    datasets = args.datasets or discover_datasets(args.data_root)
    rows = []

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        ds_root = args.data_root / dataset
        train_path = ds_root / "preprocessed_data_train.csv"
        test_path = ds_root / "preprocessed_data_test.csv"
        if not train_path.exists() or not test_path.exists():
            print(f"[skip] missing split files for {dataset}")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        test_df = pd.read_csv(test_path, sep="\t")

        seqs = get_sequences(train_df)
        train_seqs, val_seqs = split_train_val_ensure_all_items(
            seqs=seqs,
            all_items_set=set(train_df["item_id"].unique().tolist()),
            train_ratio=0.8,
            seed=args.seed,
        )
        fit_users = {uid for _, _, uid in train_seqs}
        val_users = {uid for _, _, uid in val_seqs}
        fit_df = train_df[train_df["user_id"].isin(fit_users)].copy()
        val_df = train_df[train_df["user_id"].isin(val_users)].copy()

        result = train_static_model(
            fit_df=fit_df,
            val_df=val_df,
            device=device,
            rank=args.rank,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            patience=args.patience,
            max_history=args.max_history,
        )

        ckpt_path = checkpoint_dir / f"{dataset}_static_kt.pt"
        torch.save(result.model.state_dict(), ckpt_path)

        preds, _ = predict_dataframe(
            model=result.model,
            df=test_df,
            device=device,
            max_history=args.max_history,
            chunk_size=args.batch_size,
        )
        metrics = compute_metrics(test_df["correct"].to_numpy(dtype=int), preds)
        out_df = test_df.copy()
        out_df["STATIC_KT"] = preds
        out_path = args.output_dir / f"{dataset}_static_kt_predictions.csv"
        out_df.to_csv(out_path, index=False)

        row = {"dataset": dataset, **{f"val_{k}": v for k, v in result.val_metrics.items()}, **metrics}
        rows.append(row)
        print(f"Test AUC={metrics['auc']:.4f} | saved {out_path}")

        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        metrics_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
        print(f"\nSaved metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
