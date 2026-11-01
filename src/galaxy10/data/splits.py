from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SPLITS_DIR = ROOT / "splits"
LABELS_PATH = ROOT / "data" / "processed" / "galaxy10_labels.npy"


def load_splits() -> dict[str, np.ndarray]:
    labels = np.load(LABELS_PATH, allow_pickle=False)
    n_samples = len(labels)
    splits = {}

    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{name}_idx.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        indices = np.load(path, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{path} must contain a 1D integer array")
        if len(indices) == 0:
            raise ValueError(f"{name} split is empty")
        if np.any(indices < 0) or np.any(indices >= n_samples):
            raise ValueError(f"{name} split contains out-of-range indices")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} split contains repeated indices")

        splits[name] = indices.astype(np.int64, copy=False)

    combined = np.concatenate(list(splits.values()))
    if len(np.unique(combined)) != n_samples:
        raise ValueError("Splits overlap or omit samples; expected each sample exactly once")

    return splits