"""
experiment_tau_sweep.py
=======================

Option C: tau ablation for GFL and GFL-q.

For each tau in --tau_values, runs:
    - GFL  (gamma=2.0, tau=<this>)
    - GFL-q (gamma=2.0, tau=<this>, q=0.7)

Plus the fixed baselines (CE, FL) for context, run ONCE per noise rate
(they don't depend on tau). This is much more efficient than re-running
CE/FL for every tau value.

This produces the empirical signature of the (1+tau)^-2 influence reduction
predicted by Theorem 8.2: at fixed noise rate, minority-F1 should improve
monotonically with tau until tau becomes so large the loss undertrains.

Output schema is identical to experiment.py, so the same downstream analysis
code works on it.

Usage
-----
Recommended:
    python experiment_tau_sweep.py \\
        --data_dir /mnt/c/CN/PROJECT2/CICIDS \\
        --benign_subsample 300000 \\
        --epochs 20 --n_seeds 3 \\
        --noise_rates 0.0,0.20 \\
        --tau_values 0.0,0.3,1.0,2.0,5.0 \\
        --rare_threshold 5000 \\
        --out_dir /mnt/c/CN/PROJECT2/out_tau_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from data import (
    find_benign_class,
    identify_minority_classes,
    inject_label_noise,
    load_cicids2017,
    preprocess_and_split,
)
from experiment import run_one_config, set_seed


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

    minority_post = identify_minority_classes(
        splits_clean["y_train"], class_names, threshold=args.rare_threshold,
    )
    benign_class = find_benign_class(class_names)

    print(f"\nMinority classes (< {args.rare_threshold} samples in train, BENIGN excluded):")
    for c in minority_post:
        n = int((splits_clean["y_train"] == c).sum())
        print(f"  [{c:>2}] {class_names[c]:<35s} n_train={n}")
    if benign_class is None:
        print("\n[ERROR] BENIGN class not found.")
        raise SystemExit(1)
    print(f"\nBENIGN class index: {benign_class}  "
          f"(n_train={int((splits_clean['y_train'] == benign_class).sum())})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Parse tau values from comma-separated string.
    tau_values = [float(t) for t in args.tau_values.split(",")]
    noise_rates = [float(r) for r in args.noise_rates.split(",")]

    # Build the loss config list:
    #   - CE and FL once per noise rate (no tau dependence)
    #   - GFL at each tau value
    #   - GFL-q at each tau value
    fixed_configs: List[Tuple[str, dict]] = [
        ("ce", {}),
        ("fl", {"gamma": 2.0}),
    ]
    gfl_configs: List[Tuple[str, dict]] = [
        ("gfl", {"gamma": 2.0, "tau": t}) for t in tau_values
    ]
    gfl_q_configs: List[Tuple[str, dict]] = [
        ("gfl_q", {"gamma": 2.0, "tau": t, "q": 0.7}) for t in tau_values
    ]
    all_configs = fixed_configs + gfl_configs + gfl_q_configs

    n_runs = len(all_configs) * len(noise_rates) * args.n_seeds
    print("\n" + "=" * 70)
    print(f"TAU-SWEEP GRID: {len(all_configs)} configs x {len(noise_rates)} "
          f"noise rates x {args.n_seeds} seeds = {n_runs} runs")
    print(f"Tau values: {tau_values}")
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

        for loss_name, loss_kwargs in all_configs:
            tau_str = f"_tau{loss_kwargs.get('tau', 'NA')}" if "tau" in loss_kwargs else ""
            label = f"{loss_name}{tau_str}"

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
                print(f"  noise={noise_rate:>4}  loss={label:<14}  seed={seed}  "
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

            results[f"noise_{noise_rate}_{label}"] = {
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

    out = {
        "config":                 vars(args),
        "tau_values":             tau_values,
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

    # ---------------------------------------------------------------------
    # Summary: tau curves at each noise rate
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TAU SWEEP: minority-F1 mean as function of tau")
    print("=" * 70)
    for noise_rate in noise_rates:
        print(f"\n--- noise_rate = {noise_rate} ---")
        # Baselines (no tau).
        for fixed in fixed_configs:
            label = f"{fixed[0]}"
            res = results[f"noise_{noise_rate}_{label}"]
            print(f"  {label:<10}  minority_F1 = {res['minority_f1_mean']:.4f}  "
                  f"(macro={res['macro_f1_mean']:.4f})")

        # Header for tau columns.
        header = f"  {'loss':<10}" + "".join(f"  tau={t:>5.2f}" for t in tau_values)
        print(header)
        for loss_name, kwargs_list in [("gfl", gfl_configs), ("gfl_q", gfl_q_configs)]:
            row = f"  {loss_name:<10}"
            for _, kw in kwargs_list:
                tau = kw["tau"]
                label = f"{loss_name}_tau{tau}"
                res = results[f"noise_{noise_rate}_{label}"]
                row += f"  {res['minority_f1_mean']:>7.4f}"
            print(row)

    # ---------------------------------------------------------------------
    # Theory check: at fixed noise rate, GFL minority-F1 should rise with
    # tau (up to undertraining). The (1+tau)^-2 factor is the prediction.
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("THEORY CHECK: GFL minority-F1 vs tau at noise=0.20")
    print("=" * 70)
    if 0.2 in noise_rates:
        print(f"\nTheorem 8.2 prediction: minority-F1 rises with tau via (1+tau)^-2 attenuation.")
        print(f"Observed:")
        for loss_name in ["gfl", "gfl_q"]:
            print(f"\n  {loss_name}:")
            for t in tau_values:
                label = f"{loss_name}_tau{t}"
                res = results[f"noise_0.2_{label}"]
                influence_factor = 1.0 / (1.0 + t) ** 2 if t > 0 else 1.0
                print(f"    tau={t:>5.2f}  minority_F1={res['minority_f1_mean']:.4f}  "
                      f"(theoretical (1+tau)^-2 = {influence_factor:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GFL tau-ablation experiment (Option C).",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./out_tau_sweep")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--rare_threshold", type=int, default=5000)
    parser.add_argument("--noise_rates", type=str, default="0.0,0.20",
                        help="Comma-separated noise rates. Recommended for tau sweep: "
                             "two values, the clean baseline and the high-noise condition.")
    parser.add_argument("--tau_values", type=str, default="0.0,0.3,1.0,2.0,5.0",
                        help="Comma-separated tau values to sweep. tau=0 recovers FL exactly. "
                             "Recommended: 0.0,0.3,1.0,2.0,5.0 spans the full influence range "
                             "from no saturation (0.0) to heavy saturation (5.0 -> influence "
                             "reduction factor (1+5)^-2 = 0.028).")
    parser.add_argument("--noise_mode", type=str, default="benign_to_minority",
                        choices=["benign_to_minority", "minority_to_minority"])
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--benign_subsample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    run_suite(args)


if __name__ == "__main__":
    main()
