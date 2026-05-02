"""
data.py
=======

CICIDS2017 loading + preprocessing + label-noise injection.

Targets the official MachineLearningCSV.zip distribution from
https://www.unb.ca/cic/datasets/ids-2017.html (8 CSV files).

Key design decisions for the GFL paper experiments
--------------------------------------------------
1. Label noise is injected by flipping BENIGN -> a random minority class.
   This produces the failure mode the bounded-influence theorem addresses:
   the model learns BENIGN confidently from 2M+ samples, so when the label
   says (e.g.) Heartbleed but the features look BENIGN, p_t for the label
   class is near zero -- exactly where FL's gradient diverges and GFL's
   saturates at 1/(1+tau).

2. Subsampling is BENIGN-only by default. Stratified subsampling proportionally
   starves rare classes (Heartbleed has 11 samples; a 200K stratified subsample
   leaves ~1 sample). BENIGN-only subsampling preserves all attack samples
   while cutting roughly half of the data volume (BENIGN is ~80% of CICIDS).

Public API
----------
    load_cicids2017(data_dir, benign_subsample=...)
        -> X, y, feature_names, class_names
    preprocess_and_split(X, y, ...)
        -> dict of train/val/test arrays
    inject_label_noise(y_train, X_train, y_train, ..., mode="benign_to_minority")
        -> noisy y array
    identify_minority_classes(y, class_names, threshold)
        -> list of minority class indices
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# CICIDS2017 stores web-attack labels with a literal Latin-1 0x96 (en-dash)
# character that varies between files. Normalise to plain ASCII.
_LABEL_ALIASES = {
    "Web Attack \x96 Brute Force": "Web Attack - Brute Force",
    "Web Attack \x96 XSS": "Web Attack - XSS",
    "Web Attack \x96 Sql Injection": "Web Attack - Sql Injection",
    "Web Attack \xe2\x80\x93 Brute Force": "Web Attack - Brute Force",
    "Web Attack \xe2\x80\x93 XSS": "Web Attack - XSS",
    "Web Attack \xe2\x80\x93 Sql Injection": "Web Attack - Sql Injection",
}

BENIGN_LABEL = "BENIGN"


def _normalise_label(s: str) -> str:
    s = s.strip()
    return _LABEL_ALIASES.get(s, s)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cicids2017(
    data_dir: str,
    benign_subsample: int = 0,
    max_samples: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load and concatenate all CICIDS2017 CSVs in ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory containing the 8 CICIDS2017 CSV files.
    benign_subsample : int
        If > 0, randomly subsample BENIGN to at most this many rows while
        keeping ALL attack samples intact. This is the recommended way to
        speed up runs without starving rare classes. Set 0 to disable.
    max_samples : int
        DEPRECATED -- legacy stratified subsample of the entire dataset.
        Proportionally starves rare classes (e.g. Heartbleed=11 -> ~1).
        Use benign_subsample instead. Kept for backward compatibility.
    verbose : bool
        Print progress and class distribution.

    Returns
    -------
    X : float32 array, shape (N, num_features)
    y : int64 array, shape (N,)
    feature_names : list of column names
    class_names : list of class names indexed by label encoding
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir!r}. "
            f"Download CICIDS2017 MachineLearningCSV.zip and unzip into this directory."
        )

    if verbose:
        print(f"Loading {len(csv_files)} CSV file(s) from {data_dir}...")

    dfs = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        dfs.append(df)
        if verbose:
            print(f"  {os.path.basename(path):60s} {len(df):>10d} rows")

    df = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"Concatenated: {len(df)} rows, {len(df.columns)} columns")

    if "Label" not in df.columns:
        raise KeyError(
            f"No 'Label' column found after stripping whitespace. "
            f"Columns present: {df.columns.tolist()}"
        )

    # Normalise labels (handle the Latin-1 en-dash garbage).
    df["Label"] = df["Label"].astype(str).map(_normalise_label)

    # Drop rows containing inf or NaN (CICFlowMeter artefact).
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if verbose:
        print(f"Dropped {n_before - len(df)} rows with inf/NaN values; {len(df)} remain.")

    # Drop fully-duplicate rows (well-documented CICIDS2017 issue).
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"Dropped {n_before - len(df)} duplicate rows; {len(df)} remain.")

    # ------------------------------------------------------------------
    # BENIGN-only subsampling (preferred path).
    # ------------------------------------------------------------------
    if benign_subsample > 0:
        benign_mask = (df["Label"] == BENIGN_LABEL)
        n_benign = int(benign_mask.sum())
        if n_benign > benign_subsample:
            if verbose:
                print(f"\nSubsampling BENIGN: {n_benign} -> {benign_subsample} (all attack rows kept).")
            benign_idx = df[benign_mask].sample(n=benign_subsample, random_state=42).index
            attack_idx = df[~benign_mask].index
            df = df.loc[benign_idx.union(attack_idx)].reset_index(drop=True)
            if verbose:
                print(f"After BENIGN subsample: {len(df)} rows total.")
        elif verbose:
            print(f"\nBENIGN already has only {n_benign} rows (<= {benign_subsample}); no subsample applied.")

    # Separate features (numeric only) and labels.
    y_raw = df["Label"].values
    X_df = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    # Encode labels.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw).astype(np.int64)
    class_names = encoder.classes_.tolist()

    if verbose:
        print(f"\nFinal: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
        print("Class distribution (sorted by frequency):")
        counts = np.bincount(y, minlength=len(class_names))
        order = np.argsort(-counts)
        for idx in order:
            print(f"  {class_names[idx]:35s} {counts[idx]:>10d}")

    # Legacy stratified subsample (NOT recommended -- starves rare classes).
    if max_samples > 0 and max_samples < len(X):
        if verbose:
            print(f"\n[WARNING] Legacy stratified subsample to {max_samples} rows.")
            print(f"          This proportionally starves rare classes.")
            print(f"          Prefer --benign_subsample for the same speedup without that issue.")
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=42,
        )
        if verbose:
            print(f"Subsampled: {len(X)} rows.")

    return X, y, feature_names, class_names


# ---------------------------------------------------------------------------
# Preprocessing + split
# ---------------------------------------------------------------------------
def preprocess_and_split(
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Stratified 70/10/20 split + StandardScaler fitted on train only.

    Returns
    -------
    splits : dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    class_names_out : list of class names, possibly with some dropped if they
        had fewer than 5 samples (required minimum for stratified split).
    """
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0 and test_size + val_size < 1.0):
        raise ValueError(f"invalid split sizes: test={test_size}, val={val_size}")

    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(y.max()) + 1)]
    class_names_out = list(class_names)

    # Some classes may be too small for stratified splitting (Heartbleed=11
    # would require >= 5 samples per split). Drop those rather than failing.
    counts = np.bincount(y, minlength=len(class_names))
    too_small = np.where(counts < 5)[0]
    if len(too_small) > 0:
        if verbose:
            dropped_names = [class_names[c] for c in too_small]
            print(f"Warning: dropping {len(too_small)} class(es) with < 5 samples "
                  f"to enable stratified split: {dropped_names}")
        keep = ~np.isin(y, too_small)
        X, y = X[keep], y[keep]
        # Re-label remaining classes contiguously.
        unique = np.unique(y)
        remap = {old: new for new, old in enumerate(unique)}
        y = np.array([remap[v] for v in y], dtype=np.int64)
        class_names_out = [class_names[old] for old in unique]

    # First peel off test set, then split val from train.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=seed,
    )

    # Standardise: fit on train only, apply to val/test.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    if verbose:
        print(f"Split sizes: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    return splits, class_names_out


# ---------------------------------------------------------------------------
# Label-noise injection
# ---------------------------------------------------------------------------
def inject_label_noise(
    y: np.ndarray,
    noise_rate: float,
    minority_classes: List[int],
    benign_class: Optional[int] = None,
    mode: str = "benign_to_minority",
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """Inject label noise into the training labels.

    Parameters
    ----------
    y : int array, shape (N,)
        Training labels (this function is intended to be applied to TRAIN ONLY,
        never validation or test).
    noise_rate : float in [0, 1]
        Fraction of source labels to flip.
    minority_classes : list of int
        Class indices treated as minorities. Used as flip TARGETS in
        benign_to_minority mode, or as both source and target in
        minority_to_minority mode.
    benign_class : int, optional
        Class index of BENIGN. Required for benign_to_minority mode. Used
        as the FLIP SOURCE -- a fraction of BENIGN labels are reassigned
        to a random minority class.
    mode : "benign_to_minority" or "minority_to_minority"
        - benign_to_minority (RECOMMENDED): flip BENIGN -> random minority.
          This produces confidently-wrong predictions on rare-class targets,
          which is the failure mode the bounded-influence theorem addresses.
        - minority_to_minority (legacy): flip minority -> other minority.
          The model rarely learns minorities confidently in the first place,
          so this protocol does not exercise the noise regime the theory
          addresses. Kept only for backward compatibility.
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Print summary.

    Returns
    -------
    y_noisy : int array, shape (N,), modified copy of y.
    """
    if not (0.0 <= noise_rate <= 1.0):
        raise ValueError(f"noise_rate must be in [0, 1], got {noise_rate}")
    if noise_rate == 0.0:
        return y.copy()

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()

    if mode == "benign_to_minority":
        if benign_class is None:
            raise ValueError("benign_to_minority noise mode requires benign_class to be set")
        if len(minority_classes) == 0:
            if verbose:
                print(f"  WARNING: no minority classes available; no flips applied.")
            return y_noisy

        benign_idx = np.where(y == benign_class)[0]
        n_flip = int(round(len(benign_idx) * noise_rate))
        if n_flip == 0:
            if verbose:
                print(f"  noise_rate {noise_rate}: no flips (BENIGN has {len(benign_idx)} samples).")
            return y_noisy

        flip_idx = rng.choice(benign_idx, size=n_flip, replace=False)
        new_labels = rng.choice(np.array(minority_classes), size=n_flip)
        y_noisy[flip_idx] = new_labels

        if verbose:
            print(f"  noise_rate {noise_rate} (BENIGN->minority): flipped {n_flip} of "
                  f"{len(benign_idx)} BENIGN labels into {len(minority_classes)} minority classes.")
        return y_noisy

    if mode == "minority_to_minority":
        if len(minority_classes) < 2:
            if verbose:
                print(f"  WARNING: minority_to_minority needs >= 2 minority classes, "
                      f"got {len(minority_classes)}. No flips applied.")
            return y_noisy

        n_total_flipped = 0
        for c in minority_classes:
            idx = np.where(y == c)[0]
            n_flip = int(round(len(idx) * noise_rate))
            if n_flip == 0:
                continue
            flip_idx = rng.choice(idx, size=n_flip, replace=False)
            other_minorities = np.array([m for m in minority_classes if m != c])
            new_labels = rng.choice(other_minorities, size=n_flip)
            y_noisy[flip_idx] = new_labels
            n_total_flipped += n_flip

        if verbose:
            print(f"  noise_rate {noise_rate} (minority->minority): flipped {n_total_flipped} "
                  f"labels across {len(minority_classes)} minority classes.")
        return y_noisy

    raise ValueError(f"unknown noise mode: {mode!r}. "
                     f"expected 'benign_to_minority' or 'minority_to_minority'.")


# ---------------------------------------------------------------------------
# Minority identification
# ---------------------------------------------------------------------------
def identify_minority_classes(
    y: np.ndarray,
    class_names: List[str],
    threshold: int = 5000,
    benign_label: str = BENIGN_LABEL,
) -> List[int]:
    """Return indices of classes with fewer than ``threshold`` samples,
    excluding BENIGN.

    Default threshold of 5000 separates moderate-rare attacks (Bot, PortScan,
    SSH-Patator, FTP-Patator, etc.) and extreme-rare attacks (Heartbleed,
    Infiltration) from common attacks (DoS Hulk, DDoS).
    """
    counts = np.bincount(y, minlength=len(class_names))
    minority = []
    for c in range(len(class_names)):
        if class_names[c] == benign_label:
            continue
        if counts[c] < threshold:
            minority.append(c)
    return minority


def find_benign_class(class_names: List[str], benign_label: str = BENIGN_LABEL) -> Optional[int]:
    """Return the integer index of BENIGN in class_names, or None if absent."""
    try:
        return class_names.index(benign_label)
    except ValueError:
        return None


# ===========================================================================
# Tests   --   run with: python data.py
# ===========================================================================
def _run_tests() -> None:
    """Synthetic tests for preprocessing + noise-injection logic."""
    rng = np.random.default_rng(0)

    # Build a fake imbalanced dataset; class 0 = "BENIGN", classes 3,4 = minorities.
    n_per_class = [10000, 5000, 1000, 200, 50]
    Xs, ys = [], []
    for c, n in enumerate(n_per_class):
        Xs.append(rng.standard_normal((n, 20)).astype(np.float32))
        ys.append(np.full(n, c, dtype=np.int64))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    class_names = ["BENIGN", "DDoS", "PortScan", "SSH-Patator", "Heartbleed"]

    # ---- 1. preprocess_and_split returns (splits, class_names_out).
    splits, class_names_out = preprocess_and_split(
        X, y, class_names=class_names, test_size=0.2, val_size=0.1, seed=42, verbose=False,
    )
    train_dist = np.bincount(splits["y_train"], minlength=5) / len(splits["y_train"])
    test_dist = np.bincount(splits["y_test"], minlength=5) / len(splits["y_test"])
    assert np.allclose(train_dist, test_dist, atol=0.01), \
        f"stratification broken: train {train_dist} vs test {test_dist}"
    assert class_names_out == class_names, "all classes should survive (>=5 samples each)"
    print(f"  [pass] stratified split preserves class distribution and class_names")

    # ---- 2. Standardisation: train mean ~ 0, std ~ 1.
    assert abs(splits["X_train"].mean()) < 1e-3, f"train not centred: mean={splits['X_train'].mean()}"
    assert abs(splits["X_train"].std() - 1.0) < 1e-2, f"train not standardised: std={splits['X_train'].std()}"
    print(f"  [pass] standardisation: mean={splits['X_train'].mean():.4f}, std={splits['X_train'].std():.4f}")

    # ---- 3. Minority identification excludes BENIGN.
    minority = identify_minority_classes(y, class_names, threshold=1500)
    assert 0 not in minority, "BENIGN should never be flagged as minority"
    assert minority == [2, 3, 4], f"expected [2,3,4], got {minority}"
    print(f"  [pass] minority identification (BENIGN excluded): {minority}")

    benign_class = find_benign_class(class_names)
    assert benign_class == 0
    print(f"  [pass] benign class index: {benign_class}")

    # ---- 4. noise_rate=0 is identity in both modes.
    y_clean_b = inject_label_noise(splits["y_train"], 0.0, minority,
                                    benign_class=0, mode="benign_to_minority", verbose=False)
    y_clean_m = inject_label_noise(splits["y_train"], 0.0, minority,
                                    mode="minority_to_minority", verbose=False)
    assert np.array_equal(y_clean_b, splits["y_train"])
    assert np.array_equal(y_clean_m, splits["y_train"])
    print(f"  [pass] noise_rate=0 is identity in both modes")

    # ---- 5. benign_to_minority: ONLY benign labels flip, target is in minority set.
    y_noisy = inject_label_noise(
        splits["y_train"], 0.10, minority,
        benign_class=0, mode="benign_to_minority", seed=0, verbose=False,
    )
    benign_idx = np.where(splits["y_train"] == 0)[0]
    n_benign_flipped = (y_noisy[benign_idx] != 0).sum()
    n_benign = len(benign_idx)
    flip_ratio = n_benign_flipped / n_benign
    assert 0.05 <= flip_ratio <= 0.15, \
        f"expected ~10% benign flipped, got {flip_ratio:.2%}"
    # Non-benign rows must be untouched.
    nonbenign_idx = np.where(splits["y_train"] != 0)[0]
    n_nonbenign_flipped = (y_noisy[nonbenign_idx] != splits["y_train"][nonbenign_idx]).sum()
    assert n_nonbenign_flipped == 0, \
        f"non-BENIGN rows should not be flipped, but {n_nonbenign_flipped} were"
    # Flipped labels must be in the minority set.
    flipped_mask = (y_noisy != splits["y_train"])
    assert np.all(np.isin(y_noisy[flipped_mask], minority)), \
        f"flipped labels must be in minority set {minority}"
    print(f"  [pass] benign_to_minority: {n_benign_flipped}/{n_benign} ({flip_ratio:.1%}) BENIGN flipped, "
          f"non-BENIGN untouched, targets in minority set")

    # ---- 6. minority_to_minority: ONLY minority labels flip, into the minority set.
    y_noisy2 = inject_label_noise(
        splits["y_train"], 0.10, minority,
        mode="minority_to_minority", seed=0, verbose=False,
    )
    for c in [0, 1]:  # non-minority classes
        idx = np.where(splits["y_train"] == c)[0]
        n_changed = (y_noisy2[idx] != c).sum()
        assert n_changed == 0, f"class {c} should not be flipped in minority->minority mode"
    flipped_mask2 = (y_noisy2 != splits["y_train"])
    assert np.all(np.isin(y_noisy2[flipped_mask2], minority))
    print(f"  [pass] minority_to_minority: {flipped_mask2.sum()} flips, all within minority set")

    # ---- 7. preprocess_and_split drops a class with <5 samples.
    n_per_class_tiny = [10000, 5000, 1000, 3]  # last class has only 3 samples
    Xs, ys = [], []
    for c, n in enumerate(n_per_class_tiny):
        Xs.append(rng.standard_normal((n, 20)).astype(np.float32))
        ys.append(np.full(n, c, dtype=np.int64))
    X_tiny = np.concatenate(Xs)
    y_tiny = np.concatenate(ys)
    names_tiny = ["BENIGN", "DoS", "PortScan", "Heartbleed"]
    splits_tiny, names_out = preprocess_and_split(
        X_tiny, y_tiny, class_names=names_tiny, seed=42, verbose=False,
    )
    assert "Heartbleed" not in names_out, "tiny class should be dropped"
    assert len(names_out) == 3, f"expected 3 classes after drop, got {len(names_out)}"
    print(f"  [pass] preprocess_and_split drops <5-sample class: {names_tiny} -> {names_out}")

    print("\nAll data tests passed.")


if __name__ == "__main__":
    print("Running data.py self-tests (no CICIDS data required)...\n")
    _run_tests()