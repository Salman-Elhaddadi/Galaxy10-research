import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Galaxy10_DECals.h5"
PROCESSED = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TOLERANCE_ARCSEC = 3.0


def load_metadata():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    with h5py.File(RAW_PATH, "r") as handle:
        labels = handle["ans"][:].astype(np.int64)
        ra = handle["ra"][:].astype(np.float64)
        dec = handle["dec"][:].astype(np.float64)
    return labels, ra, dec


def load_digests_and_blank_indices():
    stats_path = PROCESSED / "image_stats_v2.npz"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Expected duplicate-image hashes from the notebook at {stats_path}"
        )

    with np.load(stats_path, allow_pickle=True) as stats:
        if "digests" not in stats:
            raise KeyError("image_stats_v2.npz does not contain 'digests'")
        digests = stats["digests"].astype(str)
        blank_indices = []
        if "ch_std" in stats:
            blank_indices = np.flatnonzero(np.max(stats["ch_std"], axis=1) < 2.0).tolist()

    return digests, blank_indices


def build_group_labels(labels, digests, ra, dec):
    parent = np.arange(len(labels))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    first_index_by_digest = {}
    for index, digest in enumerate(digests):
        if digest in first_index_by_digest:
            union(index, first_index_by_digest[digest])
        else:
            first_index_by_digest[digest] = index

    valid_mask = np.isfinite(ra) & np.isfinite(dec)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size:
        ra_rad = np.deg2rad(ra[valid_indices])
        dec_rad = np.deg2rad(dec[valid_indices])
        xyz = np.column_stack(
            (
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            )
        )
        theta = np.deg2rad(TOLERANCE_ARCSEC / 3600.0)
        chord_radius = 2 * np.sin(theta / 2)
        pairs = cKDTree(xyz).query_pairs(chord_radius, output_type="ndarray")
        for i, j in pairs:
            union(int(valid_indices[i]), int(valid_indices[j]))

    return parent, find


def main():
    labels, ra, dec = load_metadata()
    digests, blank_indices = load_digests_and_blank_indices()

    if len(digests) != len(labels):
        raise ValueError(f"Found {len(digests)} hashes for {len(labels)} labels")

    parent, find = build_group_labels(labels, digests, ra, dec)

    excluded = set()
    for digest, indices in group_by_digest(digests).items():
        if len(set(labels[indices])) > 1:
            excluded.update(indices)
    excluded.update(blank_indices)
    excluded = sorted(excluded)

    keep_mask = np.ones(len(labels), dtype=bool)
    keep_mask[excluded] = False
    candidate_indices = np.flatnonzero(keep_mask)
    if candidate_indices.size == 0:
        raise ValueError("No images remain after applying the protocol exclusions")

    group_ids = np.array([find(i) for i in range(len(labels))], dtype=np.int64)[candidate_indices]
    labels_for_split = labels[candidate_indices]

    test_splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val_rel, test_rel = next(test_splitter.split(candidate_indices, labels_for_split, group_ids))
    train_val_indices = candidate_indices[train_val_rel]
    test_indices = candidate_indices[test_rel]

    val_splitter = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED)
    train_rel, val_rel = next(
        val_splitter.split(train_val_indices, labels[train_val_indices], np.array([find(i) for i in train_val_indices]))
    )
    train_indices = train_val_indices[train_rel]
    val_indices = train_val_indices[val_rel]

    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    for name, indices in split_indices.items():
        np.save(SPLITS_DIR / f"{name}_idx.npy", indices)
        class_counts = np.bincount(labels[indices], minlength=10).tolist()
        print(f"{name}: {len(indices)} images; class counts={class_counts}")

    np.save(SPLITS_DIR / "excluded_idx.npy", np.asarray(excluded, dtype=np.int64))

    all_group_ids = np.array([find(i) for i in range(len(labels))])
    split_report = {
        "seed": SEED,
        "tolerance_arcsec": TOLERANCE_ARCSEC,
        "n_images": int(len(labels)),
        "excluded_images": len(excluded),
        "split_sizes": {name: int(len(indices)) for name, indices in split_indices.items()},
        "class_counts": {
            name: np.bincount(labels[indices], minlength=10).astype(int).tolist()
            for name, indices in split_indices.items()
        },
        "group_counts": {
            name: int(np.unique(all_group_ids[indices]).size) for name, indices in split_indices.items()
        },
        "pixel_scale_share": {},
    }

    pxscale = None
    if (RAW_PATH).exists():
        with h5py.File(RAW_PATH, "r") as handle:
            if "pxscale" in handle:
                pxscale = handle["pxscale"][:].astype(np.float64)

    if pxscale is not None:
        for name, indices in split_indices.items():
            values = pxscale[indices]
            unique_levels = np.unique(values[np.isfinite(values)])
            scale_share = {}
            for level in unique_levels:
                share = float(np.mean(np.isclose(values, level, rtol=0.0, atol=1e-12)))
                scale_share[str(float(level))] = round(share, 4)
            split_report["pixel_scale_share"][name] = scale_share

    with (SPLITS_DIR / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(split_report, handle, indent=2)
        handle.write("\n")

    assert not (set(all_group_ids[train_indices]) & set(all_group_ids[val_indices]))
    assert not (set(all_group_ids[train_indices]) & set(all_group_ids[test_indices]))
    assert not (set(all_group_ids[val_indices]) & set(all_group_ids[test_indices]))

    print("Saved splits and report; no exact-duplicate or sky-pair group crosses partitions.")


def group_by_digest(digests):
    grouped = {}
    for index, digest in enumerate(digests):
        grouped.setdefault(str(digest), []).append(index)
    return grouped


if __name__ == "__main__":
    main()