from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from static_kt.common import (
    best_constant_pfa_weight,
    choose_users_by_rows,
    compute_metrics,
    discover_datasets,
    fit_pfa_and_predict,
    safe_probs,
    split_users_for_meta,
    sort_interactions,
    train_static_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a constant StaticKT/PFA ensemble.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/constant_ensemble"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prior-col", type=str, default="BASELINE")
    parser.add_argument("--pfa-col", type=str, default="LR_sscwa")
    parser.add_argument("--max-meta-rows", type=int, default=80000)
    parser.add_argument("--meta-val-ratio", type=float, default=0.2)
    parser.add_argument("--prior-rank", type=int, default=128)
    parser.add_argument("--prior-batch-size", type=int, default=2048)
    parser.add_argument("--prior-lr", type=float, default=1e-3)
    parser.add_argument("--prior-epochs", type=int, default=3)
    parser.add_argument("--prior-patience", type=int, default=2)
    parser.add_argument("--prior-max-history", type=int, default=200)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--lr-max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    datasets = args.datasets or discover_datasets(args.data_root)
    rows = []

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        ds_root = args.data_root / dataset
        train_path = ds_root / "preprocessed_data_train.csv"
        test_path = ds_root / "preprocessed_data_test.csv"
        x_path = ds_root / "X-sscwa.npz"
        if not train_path.exists() or not test_path.exists() or not x_path.exists():
            print(f"[skip] missing file(s) for {dataset}")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        test_df = pd.read_csv(test_path, sep="\t")
        if args.prior_col not in test_df.columns or args.pfa_col not in test_df.columns:
            print(f"[skip] {dataset}: {args.prior_col}/{args.pfa_col} missing")
            continue

        meta_users = choose_users_by_rows(train_df, max_rows=args.max_meta_rows, seed=args.seed)
        meta_df = train_df[train_df["user_id"].isin(meta_users)].copy()
        fit_users, val_users = split_users_for_meta(meta_df, seed=args.seed, val_ratio=args.meta_val_ratio)
        fit_df = meta_df[meta_df["user_id"].isin(fit_users)].copy()
        val_df = meta_df[meta_df["user_id"].isin(val_users)].copy()

        result = train_static_model(
            fit_df=fit_df,
            val_df=val_df,
            device=device,
            rank=args.prior_rank,
            batch_size=args.prior_batch_size,
            lr=args.prior_lr,
            num_epochs=args.prior_epochs,
            patience=args.prior_patience,
            max_history=args.prior_max_history,
        )
        pfa_val_pred = fit_pfa_and_predict(x_path, fit_df, val_df, lr_max_iter=args.lr_max_iter)
        y_val = sort_interactions(val_df)["correct"].to_numpy(dtype=int)
        alpha, meta_metrics = best_constant_pfa_weight(
            y=y_val,
            prior_pred=result.val_pred,
            pfa_pred=pfa_val_pred,
            step=args.alpha_step,
        )

        y_test = test_df["correct"].to_numpy(dtype=int)
        prior_test = test_df[args.prior_col].to_numpy(dtype=float)
        pfa_test = test_df[args.pfa_col].to_numpy(dtype=float)
        ensemble_pred = safe_probs((1.0 - alpha) * prior_test + alpha * pfa_test)

        prior_metrics = compute_metrics(y_test, prior_test)
        pfa_metrics = compute_metrics(y_test, pfa_test)
        ensemble_metrics = compute_metrics(y_test, ensemble_pred)

        out_df = test_df.copy()
        out_df["CONSTANT_PFA_WEIGHT"] = alpha
        out_df["STATIC_PFA_CONSTANT_ENSEMBLE"] = ensemble_pred
        out_df.to_csv(args.output_dir / f"{dataset}_constant_ensemble_predictions.csv", index=False)

        row = {
            "dataset": dataset,
            "pfa_weight": alpha,
            **{f"meta_{k}": v for k, v in meta_metrics.items()},
            **{f"prior_{k}": v for k, v in prior_metrics.items()},
            **{f"pfa_{k}": v for k, v in pfa_metrics.items()},
            **{f"ensemble_{k}": v for k, v in ensemble_metrics.items()},
        }
        rows.append(row)
        print(
            f"PFA weight={alpha:.3f} | "
            f"PriorKT={prior_metrics['auc']:.4f} | "
            f"PFA={pfa_metrics['auc']:.4f} | "
            f"Ensemble={ensemble_metrics['auc']:.4f}"
        )

        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        metrics = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        print(f"\nSaved metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
