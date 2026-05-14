from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .models import StaticKT


@dataclass
class InitArtifacts:
    pi: torch.Tensor
    num_items: int


@dataclass
class TrainingResult:
    model: StaticKT
    val_pred: np.ndarray
    val_metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_probs(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = safe_probs(y_prob)
    if np.unique(y_true).size == 1:
        auc = accuracy_score(y_true, (y_prob >= 0.5).astype(np.int64))
    else:
        auc = roc_auc_score(y_true, y_prob)
    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, (y_prob >= 0.5).astype(np.int64))),
        "nll": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def initialize_from_df(df: pd.DataFrame, num_items: int | None = None) -> InitArtifacts:
    if num_items is None:
        num_items = int(df["item_id"].max() + 1)

    df_first = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    global_prior = float(df_first["correct"].mean()) if len(df_first) > 0 else 0.5
    item_means = df_first.groupby("item_id")["correct"].mean().to_dict()

    pi = torch.full((num_items,), global_prior, dtype=torch.float32)
    for item_id, mean in item_means.items():
        pi[int(item_id)] = float(mean)

    return InitArtifacts(pi=pi, num_items=num_items)


def initialize_from_path(path: Path, sep: str = "\t", num_items: int | None = None) -> InitArtifacts:
    df = pd.read_csv(path, sep=sep)
    return initialize_from_df(df, num_items=num_items)


def sort_interactions(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)


def get_sequences(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    ordered = sort_interactions(df)
    seqs = []
    for uid, u_df in ordered.groupby("user_id", sort=False):
        items = u_df["item_id"].to_numpy(dtype=np.int64)
        labels = u_df["correct"].to_numpy(dtype=np.float32)
        if len(items) == 0:
            continue
        seqs.append((items.copy(), labels.copy(), int(uid)))
    return seqs


def split_train_val_ensure_all_items(
    seqs: list[tuple[np.ndarray, np.ndarray, int]],
    all_items_set: set[int],
    train_ratio: float = 0.8,
    seed: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int]], list[tuple[np.ndarray, np.ndarray, int]]]:
    rng = random.Random(seed)
    seqs = list(seqs)
    rng.shuffle(seqs)

    n_train = max(1, int(train_ratio * len(seqs)))
    train = seqs[:n_train]
    val = seqs[n_train:]
    if not val:
        val = [train.pop()]

    def items_in(seq_list: list[tuple[np.ndarray, np.ndarray, int]]) -> set[int]:
        seen: set[int] = set()
        for items, _, _ in seq_list:
            seen.update(items.tolist())
        return seen

    train_items = items_in(train)
    missing = set(all_items_set) - train_items
    if not missing:
        return train, val

    remaining_val = []
    for sample in val:
        items, _, _ = sample
        if missing.intersection(items.tolist()):
            train.append(sample)
            train_items.update(items.tolist())
            missing = set(all_items_set) - train_items
        else:
            remaining_val.append(sample)
    val = remaining_val if remaining_val else [train.pop()]
    return train, val


class BagDataset(Dataset):
    def __init__(self, seqs: list[tuple[np.ndarray, np.ndarray, int]], max_history: int = 0):
        self.samples: list[tuple[np.ndarray, np.ndarray, int, float]] = []
        self.max_history = max_history

        for items, labels, _ in seqs:
            for t in range(len(items)):
                hist_start = max(0, t - self.max_history) if self.max_history > 0 else 0
                hist_items = items[hist_start:t]
                hist_vals = (labels[hist_start:t] * 2.0) - 1.0
                self.samples.append((hist_items, hist_vals, int(items[t] + 1), float(labels[t])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int, float]:
        return self.samples[idx]


def collate_bag_fn(batch: list[tuple[np.ndarray, np.ndarray, int, float]]) -> tuple[torch.Tensor, ...]:
    h_idx_list = []
    h_val_list = []
    targets = []
    labels = []

    for hist_idx, hist_val, target, label in batch:
        h_idx_list.append(torch.tensor(hist_idx + 1, dtype=torch.long))
        h_val_list.append(torch.tensor(hist_val, dtype=torch.float32))
        targets.append(target)
        labels.append(label)

    h_idx_pad = pad_sequence(h_idx_list, batch_first=True, padding_value=0)
    h_val_pad = pad_sequence(h_val_list, batch_first=True, padding_value=0.0)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return h_idx_pad, h_val_pad, target_tensor, label_tensor


def train_one_epoch(
    model: StaticKT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float,
    use_amp: bool,
) -> float:
    model.train()
    losses = []

    for h_idx, h_val, targets, labels in tqdm(dataloader, leave=False, desc="Train"):
        h_idx = h_idx.to(model.pi.device, non_blocking=True)
        h_val = h_val.to(model.pi.device, non_blocking=True)
        targets = targets.to(model.pi.device, non_blocking=True)
        labels = labels.to(model.pi.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=model.pi.device.type, enabled=use_amp):
            logits = model(h_idx, h_val, targets)
            loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else 0.0


@torch.inference_mode()
def evaluate(model: StaticKT, dataloader: DataLoader, use_amp: bool) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    probs_all = []
    labels_all = []

    for h_idx, h_val, targets, labels in tqdm(dataloader, leave=False, desc="Eval"):
        h_idx = h_idx.to(model.pi.device, non_blocking=True)
        h_val = h_val.to(model.pi.device, non_blocking=True)
        targets = targets.to(model.pi.device, non_blocking=True)

        with torch.autocast(device_type=model.pi.device.type, enabled=use_amp):
            probs = torch.sigmoid(model(h_idx, h_val, targets))
        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(labels.numpy())

    pred = np.concatenate(probs_all) if probs_all else np.empty(0, dtype=np.float64)
    y = np.concatenate(labels_all).astype(np.int64) if labels_all else np.empty(0, dtype=np.int64)
    return pred, y, compute_metrics(y, pred) if len(y) else {"auc": 0.5, "acc": 0.0, "nll": 0.0, "brier": 0.0}


def train_static_model(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    rank: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    patience: int,
    max_history: int = 0,
    grad_clip: float = 5.0,
) -> TrainingResult:
    fit_seqs = get_sequences(fit_df)
    val_seqs = get_sequences(val_df)
    num_items = int(max(fit_df["item_id"].max(), val_df["item_id"].max()) + 1)
    init = initialize_from_df(fit_df, num_items=num_items)

    fit_ds = BagDataset(fit_seqs, max_history=max_history)
    val_ds = BagDataset(val_seqs, max_history=max_history)
    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_bag_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bag_fn)

    model = StaticKT(pi=init.pi, rank=min(rank, num_items), pad_id=0).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"

    best_auc = -np.inf
    best_state = None
    best_pred = None
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, fit_loader, optimizer, criterion, grad_clip, use_amp)
        val_pred, y_val, val_metrics = evaluate(model, val_loader, use_amp)
        print(
            f"  StaticKT epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_auc={val_metrics['auc']:.4f}"
        )
        if val_metrics["auc"] > best_auc + 1e-4:
            best_auc = val_metrics["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = val_pred.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    assert best_state is not None and best_pred is not None
    model.load_state_dict(best_state)
    return TrainingResult(model=model, val_pred=best_pred, val_metrics=compute_metrics(y_val, best_pred))


@torch.inference_mode()
def predict_dataframe(
    model: StaticKT,
    df: pd.DataFrame,
    device: torch.device,
    max_history: int,
    chunk_size: int,
    collect_mask_stats: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    working = df.reset_index(drop=True).copy()
    working["_row_id"] = np.arange(len(working), dtype=np.int64)
    ordered = working.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    preds = np.empty(len(df), dtype=np.float32)
    use_amp = device.type == "cuda"

    total_valid = 0
    total_masked_out = 0
    rows_without_valid_history = 0

    for _, user_df in ordered.groupby("user_id", sort=False):
        row_ids = user_df["_row_id"].to_numpy(dtype=np.int64)
        items = user_df["item_id"].to_numpy(dtype=np.int64)
        labels = user_df["correct"].to_numpy(dtype=np.float32)

        for start in range(0, len(items), chunk_size):
            end = min(start + chunk_size, len(items))
            h_idx_list = []
            h_val_list = []
            targets = []

            for pos in range(start, end):
                hist_start = max(0, pos - max_history) if max_history > 0 else 0
                hist_items = items[hist_start:pos] + 1
                hist_values = (labels[hist_start:pos] * 2.0) - 1.0
                h_idx_list.append(torch.tensor(hist_items, dtype=torch.long))
                h_val_list.append(torch.tensor(hist_values, dtype=torch.float32))
                targets.append(int(items[pos] + 1))

            h_idx = pad_sequence(h_idx_list, batch_first=True, padding_value=0).to(device)
            h_val = pad_sequence(h_val_list, batch_first=True, padding_value=0.0).to(device)
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

            if collect_mask_stats:
                mask = model.history_mask(h_idx, target_tensor)
                valid = h_idx != model.pad_id
                total_valid += int(valid.sum().item())
                total_masked_out += int((valid & ~mask).sum().item())
                rows_without_valid_history += int((~mask.any(dim=1)).sum().item())

            with torch.autocast(device_type=device.type, enabled=use_amp):
                probs = torch.sigmoid(model(h_idx, h_val, target_tensor)).detach().cpu().numpy()
            preds[row_ids[start:end]] = probs.astype(np.float32)

    diagnostics = {}
    if collect_mask_stats:
        diagnostics = {
            "masked_history_ratio": float(total_masked_out / total_valid) if total_valid else 0.0,
            "no_valid_history_ratio": float(rows_without_valid_history / len(df)) if len(df) else 0.0,
        }
    return preds, diagnostics


def load_q_matrix(q_path: Path, num_items: int | None = None) -> torch.Tensor:
    q_sparse = load_npz(q_path).tocsr()
    rows = q_sparse.shape[0] if num_items is None else num_items
    q_dense = np.zeros((rows, q_sparse.shape[1]), dtype=np.bool_)
    rows_to_copy = min(rows, q_sparse.shape[0])
    q_dense[:rows_to_copy, :] = q_sparse[:rows_to_copy].toarray().astype(np.bool_)
    return torch.from_numpy(q_dense)


def choose_users_by_rows(df: pd.DataFrame, max_rows: int, seed: int) -> np.ndarray:
    counts = df.groupby("user_id", sort=False).size().reset_index(name="n")
    users = counts["user_id"].to_numpy(copy=True)
    row_counts = counts["n"].to_numpy()

    if max_rows <= 0 or int(row_counts.sum()) <= max_rows:
        return np.sort(users)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(users))
    chosen = []
    total = 0
    for idx in order:
        chosen.append(users[idx])
        total += int(row_counts[idx])
        if total >= max_rows:
            break
    return np.sort(np.asarray(chosen))


def split_users_for_meta(df: pd.DataFrame, seed: int, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    users = df["user_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    split = max(1, int(round(len(users) * (1.0 - val_ratio))))
    if split >= len(users):
        split = max(1, len(users) - 1)
    return np.sort(users[:split]), np.sort(users[split:])


def fit_pfa_and_predict(
    x_path: Path,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lr_max_iter: int,
) -> np.ndarray:
    X = load_npz(x_path).tocsr()
    user_ids = np.asarray(X[:, 0].toarray()).ravel().astype(np.int64)

    def aligned_rows(target_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        ordered_target = sort_interactions(target_df).reset_index(drop=True)
        target_users = ordered_target["user_id"].unique()
        candidate_idx = np.where(np.isin(user_ids, target_users))[0]
        X_candidate = X[candidate_idx]

        cols = ["user_id", "item_id", "timestamp", "correct", "skill_id"]
        candidate_base = pd.DataFrame(np.asarray(X_candidate[:, :5].toarray()), columns=cols).astype(np.int64)
        target_base = ordered_target[cols].astype(np.int64).copy()

        candidate_base["_dup"] = candidate_base.groupby(cols).cumcount()
        target_base["_dup"] = target_base.groupby(cols).cumcount()
        candidate_base["_x_row"] = np.arange(len(candidate_base), dtype=np.int64)

        merged = target_base.merge(candidate_base, on=cols + ["_dup"], how="left", sort=False)
        if merged["_x_row"].isna().any():
            raise ValueError(f"{x_path}: could not align sparse rows with the target dataframe.")
        row_positions = merged["_x_row"].to_numpy(dtype=np.int64)
        return ordered_target, X_candidate[row_positions]

    _, X_fit = aligned_rows(fit_df)
    _, X_val = aligned_rows(val_df)

    y_fit = np.asarray(X_fit[:, 3].toarray()).ravel().astype(np.int64)
    features_fit = X_fit[:, 5:]
    features_val = X_val[:, 5:]

    clf = LogisticRegression(solver="lbfgs", max_iter=lr_max_iter)
    clf.fit(features_fit, y_fit)
    return clf.predict_proba(features_val)[:, 1]


def best_constant_pfa_weight(
    y: np.ndarray,
    prior_pred: np.ndarray,
    pfa_pred: np.ndarray,
    step: float,
) -> tuple[float, dict[str, float]]:
    best_w = 0.5
    best_nll = float("inf")
    best_metrics: dict[str, float] = {}
    n = int(round(1.0 / step))

    for i in range(n + 1):
        w = i * step
        pred = safe_probs((1.0 - w) * prior_pred + w * pfa_pred)
        metrics = compute_metrics(y, pred)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_w = w
            best_metrics = metrics
    return best_w, best_metrics


def sigmoid_weight(x: np.ndarray, x_scale: float, slope: float, bias: float) -> np.ndarray:
    z = slope * (np.asarray(x, dtype=np.float64) / x_scale) + bias
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def sigmoid_blend(
    prior_pred: np.ndarray,
    pfa_pred: np.ndarray,
    x: np.ndarray,
    x_scale: float,
    slope: float,
    bias: float,
) -> np.ndarray:
    w_pfa = sigmoid_weight(x, x_scale, slope, bias)
    return safe_probs((1.0 - w_pfa) * safe_probs(prior_pred) + w_pfa * safe_probs(pfa_pred))


def train_sigmoid_pfa_weight(
    y: np.ndarray,
    prior_pred: np.ndarray,
    pfa_pred: np.ndarray,
    x: np.ndarray,
    x_scale: float,
    init_pfa_weight: float,
    lr: float,
    steps: int,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    prior_t = torch.tensor(safe_probs(prior_pred), dtype=torch.float32, device=device)
    pfa_t = torch.tensor(safe_probs(pfa_pred), dtype=torch.float32, device=device)
    x_t = torch.tensor(x / x_scale, dtype=torch.float32, device=device)

    slope = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    if 0.0 < init_pfa_weight < 1.0:
        bias_init = np.log(init_pfa_weight / (1.0 - init_pfa_weight))
    else:
        bias_init = 0.0
    bias = torch.nn.Parameter(torch.tensor(float(bias_init), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([slope, bias], lr=lr)

    best_loss = float("inf")
    best_slope = float(slope.detach().cpu().item())
    best_bias = float(bias.detach().cpu().item())

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        w = torch.sigmoid(slope * x_t + bias)
        pred = torch.clamp((1.0 - w) * prior_t + w * pfa_t, 1e-6, 1.0 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(pred, y_t)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_slope = float(slope.detach().cpu().item())
            best_bias = float(bias.detach().cpu().item())

    pred_np = sigmoid_blend(prior_pred, pfa_pred, x, x_scale, best_slope, best_bias)
    return best_slope, best_bias, compute_metrics(y, pred_np)


def prior_interaction_counts(df: pd.DataFrame, preserve_original_order: bool) -> np.ndarray:
    working = df.reset_index(drop=True).copy()
    working["_row_id"] = np.arange(len(working), dtype=np.int64)
    ordered = working.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    ordered["_count"] = ordered.groupby("user_id").cumcount().astype(np.float64)

    if not preserve_original_order:
        return ordered["_count"].to_numpy(dtype=np.float64)

    result = np.empty(len(df), dtype=np.float64)
    result[ordered["_row_id"].to_numpy(dtype=np.int64)] = ordered["_count"].to_numpy(dtype=np.float64)
    return result


def build_item_kcs(q_path: Path, fallback_skill_ids: np.ndarray | None = None) -> tuple[list[np.ndarray], int]:
    q = load_npz(q_path).tocsr()
    num_items, num_kcs = q.shape
    item_kcs: list[np.ndarray] = []
    for item_id in range(num_items):
        start, end = q.indptr[item_id], q.indptr[item_id + 1]
        item_kcs.append(q.indices[start:end].astype(np.int64, copy=True))

    if fallback_skill_ids is not None and len(fallback_skill_ids):
        max_skill = int(np.nanmax(fallback_skill_ids))
        if max_skill + 1 > num_kcs:
            num_kcs = max_skill + 1
    return item_kcs, num_kcs


def kc_attempt_counts(
    df: pd.DataFrame,
    q_path: Path,
    preserve_original_order: bool,
) -> np.ndarray:
    working = df.reset_index(drop=True).copy()
    working["_row_id"] = np.arange(len(working), dtype=np.int64)
    ordered = working.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    all_skill_ids = ordered["skill_id"].to_numpy(dtype=np.int64)
    item_kcs, num_kcs = build_item_kcs(q_path, fallback_skill_ids=all_skill_ids)
    counts_ordered = np.empty(len(ordered), dtype=np.float64)

    for _, user_df in ordered.groupby("user_id", sort=False):
        counts = np.zeros(num_kcs, dtype=np.float64)
        items = user_df["item_id"].to_numpy(dtype=np.int64)
        skills = user_df["skill_id"].to_numpy(dtype=np.int64)
        row_positions = user_df.index.to_numpy(dtype=np.int64)

        for local_idx, (item_id, skill_id) in enumerate(zip(items, skills)):
            if 0 <= item_id < len(item_kcs) and len(item_kcs[item_id]) > 0:
                kcs = item_kcs[item_id]
            elif 0 <= skill_id < num_kcs:
                kcs = np.asarray([skill_id], dtype=np.int64)
            else:
                kcs = np.empty(0, dtype=np.int64)

            if len(kcs) == 0:
                value = 0.0
            else:
                value = float(np.mean(counts[kcs]))
                counts[kcs] += 1.0
            counts_ordered[row_positions[local_idx]] = value

    if not preserve_original_order:
        return counts_ordered

    result = np.empty(len(df), dtype=np.float64)
    result[ordered["_row_id"].to_numpy(dtype=np.int64)] = counts_ordered
    return result


def discover_datasets(data_root: Path) -> list[str]:
    datasets = []
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "preprocessed_data_train.csv").exists() and (child / "preprocessed_data_test.csv").exists():
            datasets.append(child.name)
    return datasets
