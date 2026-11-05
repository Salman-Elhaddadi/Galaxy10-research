"""Deterministic nested grouped splits and explicit exclusion isolation."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def build_groups(labels, digests, ra, dec, tolerance_arcsec=3.0):
    from scipy.spatial import cKDTree
    n = len(labels)
    if any(len(x) != n for x in (digests, ra, dec)):
        raise ValueError("Metadata lengths differ")
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    members = {}
    for i, digest in enumerate(digests):
        members.setdefault(str(digest), []).append(i)
    for group in members.values():
        for i in group[1:]:
            union(group[0], i)
    valid = np.flatnonzero(np.isfinite(ra) & np.isfinite(dec))
    r, d = np.deg2rad(ra[valid]), np.deg2rad(dec[valid])
    xyz = np.column_stack((np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)))
    radius = 2 * np.sin(np.deg2rad(tolerance_arcsec / 3600) / 2)
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    for a, b in pairs:
        union(int(valid[a]), int(valid[b]))
    conflicts = [group for group in members.values() if len(np.unique(labels[group])) > 1]
    excluded = np.array(sorted(i for group in conflicts for i in group), dtype=np.int64)
    return np.array([find(i) for i in range(n)]), excluded, conflicts, valid[pairs]


def validate_splits(splits, excluded, groups, n):
    arrays = {**splits, "excluded": np.asarray(excluded)}
    if set(splits) != {"train", "val", "test"} or len(groups) != n:
        raise ValueError("Expected train/val/test and one group per image")
    for name, indices in arrays.items():
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"{name}: expected 1D integer indices")
        if np.any(indices < 0) or np.any(indices >= n) or len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name}: repeated or out-of-range indices")
        if name != "excluded" and not len(indices):
            raise ValueError(f"{name}: empty split")
    combined = np.concatenate(list(arrays.values()))
    if len(combined) != n or not np.array_equal(np.sort(combined), np.arange(n)):
        raise ValueError("Splits/exclusions overlap or omit images")
    names = list(splits)
    for a, name in enumerate(names):
        for other in names[a+1:]:
            if np.intersect1d(groups[splits[name]], groups[splits[other]]).size:
                raise ValueError(f"Group crosses {name} and {other}")
    return {"partition_complete": True, "excluded_in_splits": 0, "groups_crossing_splits": 0}


def generate_splits(labels, groups, excluded, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold
    candidates = np.setdiff1d(np.arange(len(labels)), excluded)
    outer = StratifiedGroupKFold(10, shuffle=True, random_state=seed)
    tv, test = next(outer.split(candidates, labels[candidates], groups[candidates]))
    remaining = candidates[tv]
    inner = StratifiedGroupKFold(9, shuffle=True, random_state=seed)
    train, val = next(inner.split(remaining, labels[remaining], groups[remaining]))
    splits = {"train": np.sort(remaining[train]), "val": np.sort(remaining[val]), "test": np.sort(candidates[test])}
    validate_splits(splits, excluded, groups, len(labels))
    return splits


def load_splits(root=ROOT):
    root = Path(root)
    labels = np.load(root / "data/processed/galaxy10_labels.npy", allow_pickle=False)
    folder = root / "splits"
    splits = {name: np.load(folder / f"{name}_idx.npy", allow_pickle=False) for name in ("train", "val", "test")}
    excluded = np.load(folder / "excluded_idx.npy", allow_pickle=False)
    groups = np.load(folder / "group_ids.npy", allow_pickle=False)
    validate_splits(splits, excluded, groups, len(labels))
    return splits
