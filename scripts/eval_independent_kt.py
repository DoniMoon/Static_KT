from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from static_kt.common import (
    compute_metrics,
    discover_datasets,
    initialize_from_df,
    load_q_matrix,
    predict_dataframe,
)
from static_kt.models import IndependentKT, StaticKT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate same-KC masking with IndependentKT.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/static_kt/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/independent_kt"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-history", type=int, default=200)
    parser.add_argument("--compare-col", type=str, default="BASELINE")
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
        q_path = ds_root / "q_mat.npz"
        ckpt_path = args.checkpoint_dir / f"{dataset}_static_kt.pt"
        if not train_path.exists() or not test_path.exists() or not q_path.exists() or not ckpt_path.exists():
            print(f"[skip] missing file(s) for {dataset}")
            continue

        train_df = pd.read_csv(train_path, sep="\t")
        test_df = pd.read_csv(test_path, sep="\t")
        num_items = int(max(train_df["item_id"].max(), test_df["item_id"].max()) + 1)
        init = initialize_from_df(train_df, num_items=num_items)
        state_dict = torch.load(ckpt_path, map_location=device)
        state_rank = int(state_dict["beta_q.weight"].shape[1])

        static_model = StaticKT(pi=init.pi, rank=state_rank, pad_id=0).to(device)
        static_model.load_state_dict(state_dict)
        q_matrix = load_q_matrix(q_path, num_items=num_items)

        independent = IndependentKT(
            pi=init.pi,
            q_matrix=q_matrix,
            rank=state_rank,
            pad_id=0,
        ).to(device)
        incompatible = independent.load_state_dict(static_model.state_dict(), strict=False)
        missing = [key for key in incompatible.missing_keys if key != "q_matrix"]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected state transfer result: {incompatible}")

        static_pred, _ = predict_dataframe(
            model=static_model,
            df=test_df,
            device=device,
            max_history=args.max_history,
            chunk_size=args.chunk_size,
        )
        independent_pred, diagnostics = predict_dataframe(
            model=independent,
            df=test_df,
            device=device,
            max_history=args.max_history,
            chunk_size=args.chunk_size,
            collect_mask_stats=True,
        )

        y = test_df["correct"].to_numpy(dtype=int)
        static_metrics = compute_metrics(y, static_pred)
        independent_metrics = compute_metrics(y, independent_pred)

        out_df = test_df.copy()
        out_df["STATIC_KT"] = static_pred
        out_df["INDEPENDENT_KT"] = independent_pred
        out_df.to_csv(args.output_dir / f"{dataset}_independent_kt_predictions.csv", index=False)

        row = {
            "dataset": dataset,
            **{f"static_{k}": v for k, v in static_metrics.items()},
            **{f"independent_{k}": v for k, v in independent_metrics.items()},
            **diagnostics,
        }
        if args.compare_col in test_df.columns:
            compare_metrics = compute_metrics(y, test_df[args.compare_col].to_numpy(dtype=float))
            row.update({f"benchmark_{k}": v for k, v in compare_metrics.items()})
            row["independent_delta_vs_benchmark_auc"] = row["independent_auc"] - row["benchmark_auc"]
        row["independent_delta_vs_static_auc"] = row["independent_auc"] - row["static_auc"]
        rows.append(row)

        print(
            f"StaticKT={row['static_auc']:.4f} | "
            f"IndependentKT={row['independent_auc']:.4f} | "
            f"delta={row['independent_delta_vs_static_auc']:+.4f}"
        )

        del static_model, independent
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        metrics = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        print(f"\nSaved metrics to {args.output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
