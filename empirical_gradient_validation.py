"""
empirical_gradient_validation.py
================================

Empirical companion to synthetic_gradient_validation.py: validate the
bounded-influence theorems on REAL CICIDS2017 samples, demonstrating that
the gradient attenuation factor predicted by Theorem 5.1(c) actually
materialises on the natural distribution of (logit, label) pairs that
arises during training on imbalanced network-flow data.

Methodology
-----------
1. Train a baseline MLP for a few epochs with cross-entropy on noisy data
   (BENIGN -> minority injection). After training, the model produces a
   distribution of confidently-wrong predictions on the noisy minority
   labels -- exactly the regime the theorems address.
2. On a held-out batch, compute per-sample p_t and per-sample logit gradient
   norm under each loss config (CE, FL, GFL(tau), GFL-q(tau)).
3. Bucket samples by p_t and report mean gradient magnitude per bucket.

This is the "do the gradients actually look like this on real data" check
that complements the synthetic validation.

Output
------
  - {out_dir}/gradient_buckets.csv   (per-bucket mean gradient by loss/tau)
  - {out_dir}/per_sample_grads.csv   (full per-sample numerics)
  - Console: bucket-by-bucket summary + asymptotic ratios

Usage
-----
    python empirical_gradient_validation.py \\
        --data_dir /mnt/c/CN/PROJECT2/CICIDS \\
        --benign_subsample 200000 \\
        --warmup_epochs 5 \\
        --noise_rate 0.20 \\
        --out_dir ./out_grad_empirical
"""

from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data import (
    find_benign_class,
    identify_minority_classes,
    inject_label_noise,
    load_cicids2017,
    preprocess_and_split,
)
from losses import (
    cross_entropy_loss, focal_loss, gfl_loss, gfl_q_loss,
)
from model import MLP


def per_sample_logit_grad_norm(loss_fn, logits, targets):
    """Per-sample ||dL/dz||_2, shape (N,)."""
    z = logits.clone().detach().requires_grad_(True)
    losses = loss_fn(z, targets, reduction="none")
    grads = torch.autograd.grad(
        outputs=losses.sum(),
        inputs=z,
        retain_graph=False,
    )[0]
    return grads.norm(dim=1).detach()


def make_loss_callable(name, **kwargs):
    if name == "ce":
        return lambda z, y, reduction: cross_entropy_loss(z, y, reduction=reduction, **kwargs)
    if name == "fl":
        return lambda z, y, reduction: focal_loss(z, y, reduction=reduction, **kwargs)
    if name == "gfl":
        return lambda z, y, reduction: gfl_loss(z, y, reduction=reduction, **kwargs)
    if name == "gfl_q":
        return lambda z, y, reduction: gfl_q_loss(z, y, reduction=reduction, **kwargs)
    raise ValueError(name)


def warmup_train(model, X_train, y_train, epochs, batch_size, lr, device):
    """Train CE for a few epochs to reach a state with confident-wrong predictions."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    n = X_train.size(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            X = X_train.index_select(0, idx)
            y = y_train.index_select(0, idx)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(X)
                loss = cross_entropy_loss(logits, y, reduction="mean")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()
        print(f"  epoch {epoch+1}/{epochs} done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./out_grad_empirical")
    p.add_argument("--benign_subsample", type=int, default=200000)
    p.add_argument("--rare_threshold", type=int, default=5000)
    p.add_argument("--noise_rate", type=float, default=0.20)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--q", type=float, default=0.7)
    p.add_argument("--tau_values", type=str, default="0.3,1.0,2.0,5.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval_samples", type=int, default=20000,
                   help="Cap eval batch to keep memory low; 20K samples covers the p_t range densely.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_values = [float(t) for t in args.tau_values.split(",")]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 74)
    print("EMPIRICAL GRADIENT VALIDATION (real CICIDS samples)")
    print("=" * 74)

    # 1. Load + preprocess.
    X, y, _, class_names = load_cicids2017(
        args.data_dir, benign_subsample=args.benign_subsample,
    )
    splits, class_names = preprocess_and_split(X, y, class_names=class_names, seed=args.seed)
    num_classes = int(splits["y_train"].max()) + 1
    minority = identify_minority_classes(splits["y_train"], class_names, threshold=args.rare_threshold)
    benign_class = find_benign_class(class_names)
    print(f"\nminority indices: {minority} ({len(minority)} classes)")
    print(f"BENIGN index: {benign_class}")

    # 2. Inject noise into training labels.
    if args.noise_rate > 0:
        y_noisy = inject_label_noise(
            splits["y_train"], args.noise_rate, minority,
            benign_class=benign_class, mode="benign_to_minority", seed=0,
        )
        splits["y_train"] = y_noisy

    # 3. Move to device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    X_train = torch.from_numpy(splits["X_train"]).float().to(device)
    y_train = torch.from_numpy(splits["y_train"]).long().to(device)
    X_test  = torch.from_numpy(splits["X_test"]).float().to(device)
    y_test  = torch.from_numpy(splits["y_test"]).long().to(device)

    # 4. Train CE warmup.
    print(f"\nWarming up MLP with CE for {args.warmup_epochs} epochs...")
    input_dim = X_train.size(1)
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    t0 = time.time()
    warmup_train(model, X_train, y_train, args.warmup_epochs, args.batch_size, args.lr, device)
    print(f"Warmup done in {time.time() - t0:.1f}s.")

    # 5. Eval batch: take the test set, optionally subsampled.
    n_test = X_test.size(0)
    if n_test > args.max_eval_samples:
        idx = torch.randperm(n_test, device=device)[:args.max_eval_samples]
        X_eval = X_test.index_select(0, idx)
        y_eval = y_test.index_select(0, idx)
    else:
        X_eval, y_eval = X_test, y_test
    print(f"\nEval batch: {X_eval.size(0)} samples")

    # 6. Forward pass to get logits + p_t.
    model.eval()
    with torch.no_grad():
        logits_eval = model(X_eval).float()  # cast back from autocast
        p_t = torch.softmax(logits_eval, dim=-1).gather(
            1, y_eval.view(-1, 1)
        ).squeeze(1)
    p_t_np = p_t.cpu().numpy()
    print(f"p_t distribution: min={p_t_np.min():.2e}, "
          f"median={np.median(p_t_np):.4f}, max={p_t_np.max():.4f}")
    print(f"  fraction with p_t < 0.05 (deep tail / confidently wrong): "
          f"{(p_t_np < 0.05).mean():.3f}")
    print(f"  fraction with p_t < 0.50 (hard / mid):                    "
          f"{(p_t_np < 0.50).mean():.3f}")

    # 7. Per-sample gradient norms.
    print("\nComputing per-sample gradient norms for each loss config...")
    grads = {}
    grads["ce"] = per_sample_logit_grad_norm(
        make_loss_callable("ce"), logits_eval, y_eval
    ).cpu().numpy()
    grads["fl"] = per_sample_logit_grad_norm(
        make_loss_callable("fl", gamma=args.gamma), logits_eval, y_eval
    ).cpu().numpy()
    for tau in tau_values:
        grads[f"gfl_tau{tau}"] = per_sample_logit_grad_norm(
            make_loss_callable("gfl", gamma=args.gamma, tau=tau), logits_eval, y_eval
        ).cpu().numpy()
        grads[f"gfl_q_tau{tau}"] = per_sample_logit_grad_norm(
            make_loss_callable("gfl_q", gamma=args.gamma, tau=tau, q=args.q),
            logits_eval, y_eval
        ).cpu().numpy()

    # Per-sample CSV.
    per_sample_csv = out_dir / "per_sample_grads.csv"
    cols = ["p_t"] + list(grads.keys())
    with open(per_sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(p_t_np)):
            row = [float(p_t_np[i])] + [float(grads[c][i]) for c in grads]
            w.writerow(row)
    print(f"\nPer-sample data: {per_sample_csv}")

    # Bucket summary.
    print()
    print("=" * 74)
    print("BUCKETED MEAN |grad| ON REAL CICIDS SAMPLES")
    print("=" * 74)
    print("(values in each bucket are mean gradient norm averaged over samples")
    print(" with p_t in the indicated range)")
    print()
    buckets = [
        (0.0,    1e-3,  "p_t < 1e-3 (confidently-wrong)"),
        (1e-3,   0.05,  "p_t in [1e-3, 0.05)"),
        (0.05,   0.30,  "p_t in [0.05, 0.30)"),
        (0.30,   0.60,  "p_t in [0.30, 0.60) (mid-hard)"),
        (0.60,   0.95,  "p_t in [0.60, 0.95)"),
        (0.95,   1.01,  "p_t in [0.95, 1.0)"),
    ]
    bucket_csv = out_dir / "gradient_buckets.csv"
    bucket_rows = []
    print(f"{'bucket':<35}  {'n':>6}  {'CE':>8}  {'FL':>8}  " +
          "  ".join(f"{f'GFL t={t}':>10}" for t in tau_values))
    for low, high, name in buckets:
        mask = (p_t_np >= low) & (p_t_np < high)
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<35}  {n:>6}  (no samples)")
            continue
        ce_m = float(grads["ce"][mask].mean())
        fl_m = float(grads["fl"][mask].mean())
        row = f"{name:<35}  {n:>6}  {ce_m:>8.4f}  {fl_m:>8.4f}"
        bucket_row = {"bucket": name, "n": n, "ce": ce_m, "fl": fl_m}
        for tau in tau_values:
            v = float(grads[f"gfl_tau{tau}"][mask].mean())
            row += f"  {v:>10.4f}"
            bucket_row[f"gfl_tau{tau}"] = v
        for tau in tau_values:
            bucket_row[f"gfl_q_tau{tau}"] = float(grads[f"gfl_q_tau{tau}"][mask].mean())
        print(row)
        bucket_rows.append(bucket_row)

    # Save bucket CSV
    if bucket_rows:
        with open(bucket_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
            w.writeheader()
            for r in bucket_rows:
                w.writerow(r)
        print(f"\nBucketed data: {bucket_csv}")

    # Asymptotic ratio check on real data.
    print()
    print("=" * 74)
    print("THEOREM 5.1(c) ON REAL DATA: ratio |grad GFL| / |grad FL| at low p_t")
    print("=" * 74)
    print("Theory: ratio -> 1/(1+tau) as p_t -> 0+")
    deep = (p_t_np < 0.05)
    if deep.any():
        n_deep = int(deep.sum())
        print(f"({n_deep} real samples with p_t < 0.05)")
        print(f"\n{'tau':>6}  {'predicted':>12}  {'observed':>12}  {'rel error':>12}")
        for tau in tau_values:
            pred = 1.0 / (1.0 + tau)
            ratio = grads[f"gfl_tau{tau}"][deep] / np.maximum(grads["fl"][deep], 1e-12)
            obs = float(np.mean(ratio))
            err = abs(obs - pred) / max(pred, 1e-9)
            print(f"{tau:>6.2f}  {pred:>12.4f}  {obs:>12.4f}  {err:>11.2%}")
    else:
        print("No samples with p_t < 0.05 in eval set -- the model didn't")
        print("produce confidently-wrong predictions during warmup. Try a")
        print("longer warmup or a higher noise rate.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
