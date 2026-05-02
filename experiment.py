"""
experiment.py
=============

Main driver for the GFL experiment suite on CICIDS2017.

Runs the loss x noise-rate cross-product with multiple seeds:

    losses      = {ce, fl, gce, mae, gfl, gfl_q}
    noise_rates = {0.0, 0.05, 0.10, 0.20}   (BENIGN -> minority flips)
    seeds       = 0, 1, ..., n_seeds - 1

Training is GPU-resident: the entire dataset is pushed to VRAM once before
the training loop, and minibatches are produced by index slicing rather than
by a CPU DataLoader. This eliminates the CPU/PCIe bottleneck that leaves a
modern GPU mostly idle on small tabular MLPs.

Saves results to <out_dir>/results.json and prints summary tables.

Usage
-----
Recommended (overnight) run:
    python experiment.py --data_dir /path/to/CICIDS2017/ \\
        --benign_subsample 300000 --epochs 20 --n_seeds 3 \\
        --rare_threshold 5000 --noise_rates 0.0,0.05,0.10,0.20

Quick smoke test:
    python experiment.py --data_dir /path/to/CICIDS2017/ \\
        --benign_subsample 200000 --epochs 5 --n_seeds 1 \\
        --noise_rates 0.0,0.20
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import (
    find_benign_class,
    identify_minority_classes,
    inject_label_noise,
    load_cicids2017,
    preprocess_and_split,
)
from losses import make_loss
from model import MLP


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# VRAM-resident training: entire dataset stays on the GPU; batches are produced
# by index slicing. This pegs GPU utilisation on small tabular MLPs where
# the per-batch compute is microseconds and CPU/PCIe transfer would dominate.
# ---------------------------------------------------------------------------
def _to_device_tensors(
    splits: Dict[str, np.ndarray], device: str
) -> Dict[str, torch.Tensor]:
    """Move all split arrays to the target device once, as tensors."""
    return {
        "X_train": torch.from_numpy(splits["X_train"]).float().to(device),
        "y_train": torch.from_numpy(splits["y_train"]).long().to(device),
        "X_val":   torch.from_numpy(splits["X_val"]).float().to(device),
        "y_val":   torch.from_numpy(splits["y_val"]).long().to(device),
        "X_test":  torch.from_numpy(splits["X_test"]).float().to(device),
        "y_test":  torch.from_numpy(splits["y_test"]).long().to(device),
    }


def train_one_epoch(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    batch_size: int,
    alpha=None,
) -> float:
    """One epoch of SGD over GPU-resident tensors with shuffled index slicing."""
    model.train()
    n = X_train.size(0)
    perm = torch.randperm(n, device=device)

    total_loss, count = 0.0, 0
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        X = X_train.index_select(0, idx)
        y = y_train.index_select(0, idx)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(X)
            loss = loss_fn(logits, y, alpha=alpha)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X.size(0)
        count += X.size(0)
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    eval_batch_size: int = 16384,
) -> Dict:
    """Evaluate over GPU-resident tensors. Big eval batch is fine -- no grads."""
    model.eval()
    preds_chunks = []
    n = X.size(0)
    for start in range(0, n, eval_batch_size):
        chunk = X[start:start + eval_batch_size]
        logits = model(chunk)
        preds_chunks.append(logits.argmax(dim=-1))
    preds = torch.cat(preds_chunks).cpu().numpy()
    targets = y.cpu().numpy()

    labels = list(range(num_classes))
    return {
        "macro_f1":     float(f1_score(targets, preds, average="macro",    zero_division=0)),
        "weighted_f1":  float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(targets, preds, average=None, labels=labels, zero_division=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Single config: train one (loss, seed, noise_rate) combination end-to-end.
# ---------------------------------------------------------------------------
def run_one_config(
    splits_np: Dict[str, np.ndarray],
    num_classes: int,
    loss_name: str,
    loss_kwargs: dict,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    use_class_weights: bool,
) -> Dict:
    set_seed(seed)

    # Move the entire dataset to the device once.
    tensors = _to_device_tensors(splits_np, device)
    input_dim = tensors["X_train"].size(1)

    # Optional inverse-sqrt-frequency class weights (capped via mean-norm).
    alpha = None
    if use_class_weights:
        counts = np.bincount(splits_np["y_train"], minlength=num_classes).astype(np.float32)
        weights = 1.0 / np.sqrt(counts + 1.0)
        weights = weights / weights.mean()
        alpha = torch.tensor(weights, dtype=torch.float32, device=device)

    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    loss_fn = make_loss(loss_name, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_f1 = -1.0
    best_test_metrics = None
    for epoch in range(epochs):
        train_one_epoch(
            model, tensors["X_train"], tensors["y_train"],
            loss_fn, optimizer, scaler, device, batch_size, alpha=alpha,
        )
        val_metrics = evaluate(model, tensors["X_val"], tensors["y_val"], num_classes)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_test_metrics = evaluate(model, tensors["X_test"], tensors["y_test"], num_classes)
        scheduler.step()

    # Free the GPU copy before the next config builds its own.
    del tensors
    if device == "cuda":
        torch.cuda.empty_cache()

    return best_test_metrics


# ---------------------------------------------------------------------------
# Full experiment suite
# ---------------------------------------------------------------------------
def run_suite(args) -> None:
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    X, y, feature_names, class_names = load_cicids2017(
        args.data_dir,
        benign_subsample=args.benign_subsample,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING + SPLIT")
    print("=" * 70)
    splits_clean, class_names = preprocess_and_split(
        X, y, class_names=class_names, seed=args.split_seed,
    )
    num_classes = int(splits_clean["y_train"].max()) + 1

    # Identify minorities and BENIGN AFTER class-name remapping in preprocess.
    minority_post = identify_minority_classes(
        splits_clean["y_train"], class_names, threshold=args.rare_threshold,
    )
    benign_class = find_benign_class(class_names)

    print(f"\nMinority classes (< {args.rare_threshold} samples in train, BENIGN excluded):")
    for c in minority_post:
        n = int((splits_clean["y_train"] == c).sum())
        print(f"  [{c:>2}] {class_names[c]:<35s} n_train={n}")
    if benign_class is None:
        print("\n[ERROR] BENIGN class not found in class_names. "
              "BENIGN->minority noise injection cannot proceed.")
        raise SystemExit(1)
    print(f"\nBENIGN class index: {benign_class}  "
          f"(n_train={int((splits_clean['y_train'] == benign_class).sum())})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        n_floats = sum(splits_clean[k].size for k in ("X_train", "X_val", "X_test")) \
                 + sum(splits_clean[k].size for k in ("y_train", "y_val", "y_test"))
        approx_gb = n_floats * 4 / 1e9
        print(f"Approx. dataset VRAM footprint: {approx_gb:.2f} GB (X+y, float32+int64)")

    # Loss configurations to compare.
    loss_configs: List[Tuple[str, dict]] = [
        ("ce",    {}),
        ("fl",    {"gamma": 2.0}),
        ("gce",   {"q": 0.7}),
        ("mae",   {}),
        ("gfl",   {"gamma": 2.0, "tau": 0.3}),
        ("gfl_q", {"gamma": 2.0, "tau": 0.3, "q": 0.7}),
    ]

    noise_rates = [float(r) for r in args.noise_rates.split(",")]

    n_runs = len(loss_configs) * len(noise_rates) * args.n_seeds
    print("\n" + "=" * 70)
    print(f"EXPERIMENT GRID: {len(loss_configs)} losses x {len(noise_rates)} "
          f"noise rates x {args.n_seeds} seeds = {n_runs} runs")
    print(f"Noise mode: {args.noise_mode}")
    print("=" * 70)

    results = {}

    for noise_rate in noise_rates:
        if noise_rate > 0.0:
            print(f"\n--- noise_rate = {noise_rate} ---")
            y_noisy = inject_label_noise(
                splits_clean["y_train"], noise_rate, minority_post,
                benign_class=benign_class, mode=args.noise_mode,
                seed=args.noise_seed,
            )
            splits = deepcopy(splits_clean)
            splits["y_train"] = y_noisy
        else:
            print(f"\n--- noise_rate = 0.0 (clean labels) ---")
            splits = splits_clean

        for loss_name, loss_kwargs in loss_configs:
            seed_results = []
            for seed in range(args.n_seeds):
                t0 = time.time()
                metrics = run_one_config(
                    splits_np=splits,
                    num_classes=num_classes,
                    loss_name=loss_name,
                    loss_kwargs=loss_kwargs,
                    seed=seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    device=device,
                    use_class_weights=args.use_class_weights,
                )
                t1 = time.time()
                seed_results.append(metrics)
                print(f"  noise={noise_rate:>4}  loss={loss_name:>6}  seed={seed}  "
                      f"macro_f1={metrics['macro_f1']:.4f}  "
                      f"weighted_f1={metrics['weighted_f1']:.4f}  "
                      f"({t1 - t0:.1f}s)")

            macro_f1s     = np.array([r["macro_f1"]    for r in seed_results])
            weighted_f1s  = np.array([r["weighted_f1"] for r in seed_results])
            per_class_f1s = np.array([r["per_class_f1"] for r in seed_results])
            minority_f1_mean = (
                float(per_class_f1s[:, minority_post].mean())
                if minority_post else 0.0
            )

            results[f"noise_{noise_rate}_loss_{loss_name}"] = {
                "noise_rate":         noise_rate,
                "loss":               loss_name,
                "loss_kwargs":        loss_kwargs,
                "macro_f1_mean":      float(macro_f1s.mean()),
                "macro_f1_std":       float(macro_f1s.std()),
                "weighted_f1_mean":   float(weighted_f1s.mean()),
                "weighted_f1_std":    float(weighted_f1s.std()),
                "per_class_f1_mean":  per_class_f1s.mean(axis=0).tolist(),
                "per_class_f1_std":   per_class_f1s.std(axis=0).tolist(),
                "minority_f1_mean":   minority_f1_mean,
            }

    # ---------------------------------------------------------------------
    # Save and summarise
    # ---------------------------------------------------------------------
    out = {
        "config":                 vars(args),
        "class_names":            class_names,
        "minority_class_indices": minority_post,
        "minority_class_names":   [class_names[c] for c in minority_post],
        "benign_class_index":     benign_class,
        "results":                results,
    }
    out_path = Path(args.out_dir) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary tables.
    print("\n" + "=" * 70)
    print("SUMMARY: macro-F1 mean +/- std  by  (loss, noise_rate)")
    print("=" * 70)
    header = f"{'loss':<8}" + "".join(f"{'noise=' + str(r):>18}" for r in noise_rates)
    print(header)
    print("-" * len(header))
    for loss_name, _ in loss_configs:
        row = f"{loss_name:<8}"
        for r in noise_rates:
            res = results[f"noise_{r}_loss_{loss_name}"]
            row += f"  {res['macro_f1_mean']:.3f} +/- {res['macro_f1_std']:.3f}"
        print(row)

    print("\n" + "=" * 70)
    print("SUMMARY: minority-class F1 mean  by  (loss, noise_rate)")
    print("=" * 70)
    header = f"{'loss':<8}" + "".join(f"{'noise=' + str(r):>14}" for r in noise_rates)
    print(header)
    print("-" * len(header))
    for loss_name, _ in loss_configs:
        row = f"{loss_name:<8}"
        for r in noise_rates:
            res = results[f"noise_{r}_loss_{loss_name}"]
            row += f"        {res['minority_f1_mean']:.4f}"
        print(row)

    # Bounded-influence prediction check.
    print("\n" + "=" * 70)
    print("BOUNDED-INFLUENCE CHECK: minority-F1 degradation slope")
    print("=" * 70)
    print(f"For each loss, delta = (minority-F1 at max noise) - (minority-F1 at zero noise)")
    print(f"More-negative delta = worse degradation under noise.\n")
    for loss_name, _ in loss_configs:
        f1_at_zero = results[f"noise_{noise_rates[0]}_loss_{loss_name}"]["minority_f1_mean"]
        f1_at_max  = results[f"noise_{noise_rates[-1]}_loss_{loss_name}"]["minority_f1_mean"]
        delta = f1_at_max - f1_at_zero
        print(f"  {loss_name:<8}  delta = {delta:+.4f}   (zero={f1_at_zero:.4f} -> max={f1_at_max:.4f})")
    print(f"\nTheory predicts: |delta_GFL| < |delta_FL|, and |delta_GFL_q| < |delta_GFL|.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GFL experiment driver for CICIDS2017.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing CICIDS2017 CSV files.")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Training batch size. With VRAM-resident data, 2048 keeps the GPU "
                             "saturated while preserving rare-class representation per batch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--rare_threshold", type=int, default=5000,
                        help="Train classes with fewer samples are 'minority' (subject to noise "
                             "injection target and minority-F1 reporting). Default 5000 separates "
                             "moderate-rare attacks from common ones; BENIGN is always excluded.")
    parser.add_argument("--noise_rates", type=str, default="0.0,0.05,0.10,0.20",
                        help="Comma-separated noise rates to sweep.")
    parser.add_argument("--noise_mode", type=str, default="benign_to_minority",
                        choices=["benign_to_minority", "minority_to_minority"],
                        help="Label-noise protocol. benign_to_minority (default) flips a fraction "
                             "of BENIGN labels to a random minority class -- this produces "
                             "confidently-wrong predictions that exercise the bounded-influence "
                             "regime. minority_to_minority is the legacy protocol and is not "
                             "recommended for theory-mapped experiments.")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Apply inverse-sqrt-frequency class weights via the alpha argument.")
    parser.add_argument("--benign_subsample", type=int, default=0,
                        help="If > 0, subsample BENIGN to this many rows (all attack rows kept). "
                             "Recommended for fast iteration: BENIGN is ~80%% of CICIDS so this "
                             "halves dataset size without starving rare classes.")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="DEPRECATED. Stratified subsample of the full dataset, which "
                             "proportionally starves rare classes. Use --benign_subsample.")
    args = parser.parse_args()

    run_suite(args)


if __name__ == "__main__":
    main()