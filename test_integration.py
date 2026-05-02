"""
test_integration.py
===================

End-to-end integration test that synthesises a CICIDS2017-like CSV and runs
the full pipeline (load -> preprocess -> noise inject -> train -> evaluate)
for ONE loss config and ONE noise rate.

Verifies the entire stack works without requiring the real dataset.
Run: python test_integration.py
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import (
    BENIGN_LABEL,
    find_benign_class,
    identify_minority_classes,
    inject_label_noise,
    load_cicids2017,
    preprocess_and_split,
)
from experiment import run_one_config


def synth_cicids_csv(path: str, n_per_class: dict, n_features: int = 20) -> None:
    """Write a fake CICIDS-style CSV with the given per-class sample counts."""
    rng = np.random.default_rng(0)
    parts = []
    for cls, n in n_per_class.items():
        # Class-specific feature distribution to make classes learnable.
        center = rng.standard_normal(n_features) * 2.0
        feats = rng.standard_normal((n, n_features)) + center
        df = pd.DataFrame(feats, columns=[f"Feat_{i}" for i in range(n_features)])
        df["Label"] = cls
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0)
    # Add columns / values that exercise the cleaning logic.
    out[" Junk Whitespace Col "] = rng.standard_normal(len(out))
    out.iloc[0, 0] = np.inf
    out.iloc[1, 1] = -np.inf
    out.iloc[2, 2] = np.nan
    out.to_csv(path, index=False)


def main() -> None:
    print("Running integration test...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Synthetic CICIDS-like dataset with realistic-ish imbalance.
        synth_cicids_csv(
            os.path.join(tmpdir, "Wednesday-fake.csv"),
            n_per_class={
                BENIGN_LABEL:     20000,
                "DoS Hulk":        5000,
                "PortScan":        3000,
                "DDoS":            2000,
                "FTP-Patator":      500,
                "Infiltration":      50,
                "Heartbleed":        20,
            },
            n_features=20,
        )

        # 1. Load.
        print("[1] Loading synthetic data...")
        X, y, feature_names, class_names = load_cicids2017(tmpdir, verbose=False)
        assert X.dtype == np.float32 and y.dtype == np.int64
        assert X.shape[1] == 21  # 20 numeric + 1 junk whitespace col
        assert BENIGN_LABEL in class_names
        print(f"    loaded X{X.shape}, y{y.shape}, {len(class_names)} classes")

        # 2. Preprocess + split.
        print("[2] Preprocessing + split...")
        splits, class_names = preprocess_and_split(
            X, y, class_names=class_names, seed=42, verbose=False,
        )
        num_classes = int(splits["y_train"].max()) + 1
        print(f"    train={len(splits['X_train'])} val={len(splits['X_val'])} "
              f"test={len(splits['X_test'])}, K={num_classes}")

        # 3. Identify minorities + BENIGN.
        minority = identify_minority_classes(
            splits["y_train"], class_names, threshold=1000,
        )
        benign_class = find_benign_class(class_names)
        print(f"[3] Minority indices: {minority} "
              f"({[class_names[c] for c in minority]})")
        print(f"    BENIGN index: {benign_class}")
        assert benign_class is not None
        assert benign_class not in minority, "BENIGN must never appear in minority list"

        # 4. Inject BENIGN -> minority noise.
        print("[4] Injecting 10% BENIGN -> minority label noise...")
        y_noisy = inject_label_noise(
            splits["y_train"], 0.10, minority,
            benign_class=benign_class, mode="benign_to_minority",
            seed=0, verbose=False,
        )
        n_changed = int((y_noisy != splits["y_train"]).sum())
        print(f"    flipped {n_changed} labels (all from BENIGN -> minority)")

        # Verify only BENIGN rows flipped.
        nonbenign_idx = np.where(splits["y_train"] != benign_class)[0]
        n_nonbenign_flipped = int(
            (y_noisy[nonbenign_idx] != splits["y_train"][nonbenign_idx]).sum()
        )
        assert n_nonbenign_flipped == 0, \
            f"non-BENIGN rows should not flip, but {n_nonbenign_flipped} did"

        # 5. Train.
        print("[5] Training MLP with GFL for 3 epochs (CPU, VRAM-resident path)...")
        splits_noisy = dict(splits)
        splits_noisy["y_train"] = y_noisy
        metrics = run_one_config(
            splits_np=splits_noisy,
            num_classes=num_classes,
            loss_name="gfl",
            loss_kwargs={"gamma": 2.0, "tau": 0.3},
            seed=0,
            epochs=3,
            lr=1e-3,
            batch_size=128,
            device="cpu",
            use_class_weights=False,
        )
        print(f"    test macro-F1   = {metrics['macro_f1']:.4f}")
        print(f"    test weighted-F1= {metrics['weighted_f1']:.4f}")
        print(f"    per-class F1    = {[f'{x:.3f}' for x in metrics['per_class_f1']]}")

        # Sanity assertions.
        assert metrics["weighted_f1"] > 0.5, \
            f"weighted F1 {metrics['weighted_f1']:.3f} too low -- pipeline likely broken"
        assert all(0.0 <= f <= 1.0 for f in metrics["per_class_f1"]), \
            f"per-class F1 out of range: {metrics['per_class_f1']}"
        print("\n[pass] integration test successful.")


if __name__ == "__main__":
    main()