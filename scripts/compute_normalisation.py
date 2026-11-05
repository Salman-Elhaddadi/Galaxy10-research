"""Training-only normalization with split provenance and preserved previous files."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from galaxy10.audit import archive_files, sha256, write_json


def compute(root):
    processed = root / "data/processed"
    images_path = processed / "galaxy10_128x128_uint8.npy"
    train_path = root / "splits/train_idx.npy"
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    indices = np.load(train_path, allow_pickle=False)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(images)):
        raise ValueError("Invalid training indices")
    excluded = np.load(root / "splits/excluded_idx.npy", allow_pickle=False)
    val = np.load(root / "splits/val_idx.npy", allow_pickle=False)
    groups = np.load(root / "splits/group_ids.npy", allow_pickle=False)
    if np.intersect1d(indices, excluded).size or np.intersect1d(indices, val).size or np.intersect1d(groups[indices], groups[val]).size:
        raise ValueError("Training indices violate exclusion/development group isolation")
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(indices), 128):
        batch = images[indices[start:start+128]].astype(np.float64) / 255
        sums += batch.sum((0, 1, 2))
        squares += np.square(batch).sum((0, 1, 2))
        count += np.prod(batch.shape[:3]).item()
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    if (std <= 0).any():
        raise ValueError("Nonpositive normalization standard deviation")
    archive_files(processed, ["train_normalization.npz", "normalization.json"])
    np.savez(processed / "train_normalization.npz", mean=mean, std=std)
    write_json(processed / "normalization.json", {
        "schema_version": 1, "train_indices_sha256": sha256(train_path),
        "images_sha256": sha256(images_path), "normalization_sha256": sha256(processed / "train_normalization.npz"),
        "n_train": len(indices), "mean": mean.tolist(), "std": std.tolist(), "test_access": False})
    print("Training-only normalization:", mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    compute(parser.parse_args().root)
