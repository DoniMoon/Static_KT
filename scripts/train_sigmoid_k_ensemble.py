from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from static_kt.common import (
    best_constant_pfa_weight,
    choose_users_by_rows,
    compute_metrics,
    discover_datasets,
    fit_pfa_and_predict,
    prior_interaction_counts,
    safe_probs,
    sigmoid_blend,
    sigmoid_weight,
    split_users_for_meta,
    sort_interactions,
    train_sigmoid_pfa_weight,
    train_static_model,
)


def plot_sigmoids(results: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    axes = axes.flatten()

    for ax, row in zip(axes, results.itertuples(index=False)):
        x_max = max(1.0, float(row.test_k_p99))
        x = np.linspace(0.0, x_max, 300)
        y = sigmoid_weight(x, row.k_scale, row.sigmoid_slope, row.sigmoid_bias)
        ax.plot(x, y, color="#1f77b4", linewidth=2.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{row.dataset}\nAUC={row.sigmoid_auc:.3f}")
        ax.set_xlabel("k = previous user interactions")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    for ax in axes[len(results) :]:
        ax.axis("off")

    axes[0].set_ylabel("PFA weight")
    axes[3].set_ylabel("PFA weight")
    axes[6].set_ylabel("PFA weight")
    fig.suptitle("Sigmoid Ensemble by User Interaction Count", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "sigmoid_k_curves.png", dpi=200)
    fig.savefig(out_dir / "sigmoid_k_curves.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a k-dependent sigmoid StaticKT/PFA ensemble.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sigmoid_k_ensemble"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prior-col", type=str, default="BASELINE")
    parser.add_argument("--pfa-col", type=str, default="LR_sscwa")
    parser.add_argument("--max-meta-rows", type=int, default=120000)
    parser.add_argument("--meta-val-ratio", type=float, default=0.2)
    parser.add_argument("--prior-rank", type=int, default=128)
    parser.add_argument("--prior-batch-size", type=int, default=2048)
    parser.add_argument("--prior-lr", type=float, default=1e-3)
    parser.add_argument("--prior-epochs", type=int, default=3)
    parser.add_argument("--prior-patience", type=int, default=2)
    parser.add_argument("--prior-max-history", type=int, default=200)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--sigmoid-lr", type=float, default=0.05)
    parser.add_argument("--sigmoid-steps", type=int, default=3000)
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
        k_val = prior_interaction_counts(val_df, preserve_original_order=False)
        k_scale = max(1.0, float(np.percentile(k_val, 95)))

        const_w, _ = best_constant_pfa_weight(y_val, result.val_pred, pfa_val_pred, step=args.alpha_step)
        slope, bias, _ = train_sigmoid_pfa_weight(
            y=y_val,
            prior_pred=result.val_pred,
            pfa_pred=pfa_val_pred,
            x=k_val,
            x_scale=k_scale,
            init_pfa_weight=min(max(const_w, 1e-4), 1 - 1e-4),
            lr=args.sigmoid_lr,
            steps=args.sigmoid_steps,
            device=device,
        )

        y_test = test_df["correct"].to_numpy(dtype=int)
        prior_test = test_df[args.prior_col].to_numpy(dtype=float)
        pfa_test = test_df[args.pfa_col].to_numpy(dtype=float)
        k_test = prior_interaction_counts(test_df, preserve_original_order=True)
        pred = sigmoid_blend(prior_test, pfa_test, k_test, k_scale, slope, bias)
        metrics = compute_metrics(y_test, pred)
        k_p99 = float(np.percentile(k_test, 99))

        out_df = test_df.copy()
        out_df["k_prior_interactions"] = k_test
        out_df["PFA_WEIGHT_SIGMOID_K"] = sigmoid_weight(k_test, k_scale, slope, bias)
        out_df["SIGMOID_K_ENSEMBLE"] = pred
        out_df.to_csv(args.output_dir / f"{dataset}_sigmoid_k_predictions.csv", index=False)

        rows.append(
            {
                "dataset": dataset,
                "k_scale": k_scale,
                "test_k_p99": k_p99,
                "sigmoid_slope": slope,
                "sigmoid_bias": bias,
                "pfa_weight_at_k0": float(sigmoid_weight(np.array([0.0]), k_scale, slope, bias)[0]),
                "pfa_weight_at_kp99": float(sigmoid_weight(np.array([k_p99]), k_scale, slope, bias)[0]),
                **metrics,
            }
        )
        print(f"Sigmoid-k AUC={metrics['auc']:.4f}")

        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        metrics_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        metrics_df.to_csv(args.output_dir / "metrics.csv", index=False)
        plot_sigmoids(metrics_df, args.output_dir)
        print(f"\nSaved metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
