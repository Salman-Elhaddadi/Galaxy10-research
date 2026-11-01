from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Galaxy10_DECals.h5"
SPLITS_DIR = ROOT / "splits"
TOLERANCE_ARCSEC = 3.0

with h5py.File(RAW_PATH, "r") as f:
    ra = f["ra"][:]
    dec = f["dec"][:]
    labels = f["ans"][:].astype(np.int64)

split_name = np.full(len(labels), "", dtype="<U5")
for name in ("train", "val", "test"):
    indices = np.load(SPLITS_DIR / f"{name}_idx.npy")
    split_name[indices] = name

valid = np.isfinite(ra) & np.isfinite(dec)
indices = np.flatnonzero(valid)

# Convert sky coordinates to unit-sphere Cartesian coordinates.
ra_rad = np.deg2rad(ra[valid])
dec_rad = np.deg2rad(dec[valid])
xyz = np.column_stack((
    np.cos(dec_rad) * np.cos(ra_rad),
    np.cos(dec_rad) * np.sin(ra_rad),
    np.sin(dec_rad),
))

theta = np.deg2rad(TOLERANCE_ARCSEC / 3600.0)
chord_radius = 2 * np.sin(theta / 2)
pairs = cKDTree(xyz).query_pairs(chord_radius, output_type="ndarray")

cross_split = [
    (indices[i], indices[j])
    for i, j in pairs
    if split_name[indices[i]] != split_name[indices[j]]
]

print(f"Pairs within {TOLERANCE_ARCSEC} arcsec: {len(pairs)}")
print(f"Pairs crossing splits: {len(cross_split)}")

for i, j in cross_split[:20]:
    print(
        f"{i} ({split_name[i]}, class {labels[i]}) <-> "
        f"{j} ({split_name[j]}, class {labels[j]})"
    )