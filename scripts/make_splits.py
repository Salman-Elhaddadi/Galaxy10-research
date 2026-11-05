"""Recompute raw image digests, apply protocol policies, archive and audit splits."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import sklearn
import yaml
from galaxy10.audit import archive_files, sha256, write_json
from galaxy10.data.splits import build_groups, generate_splits, validate_splits


def prepare(root):
    root = Path(root)
    data_config = yaml.safe_load((root / "configs/data.yaml").read_text(encoding="utf-8"))
    defaults = {"split_seed": 42, "spatial_match_tolerance_arcsec": 3.0, "outer_folds": 10, "inner_folds": 9, "selected_fold": 0}
    if any(data_config.get(k) != value for k, value in defaults.items()):
        raise ValueError("Data configuration conflicts with protocol defaults; amend explicitly before regenerating")
    raw = root / "data/raw/Galaxy10_DECals.h5"
    folder = root / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "r") as handle:
        labels = handle["ans"][:].astype(np.int64)
        ra, dec, scale = (handle[name][:].astype(np.float64) for name in ("ra", "dec", "pxscale"))
        if not np.array_equal(labels, np.load(root / "data/processed/galaxy10_labels.npy", allow_pickle=False)):
            raise ValueError("Raw/cache labels differ")
        n = len(labels)
        digests = np.empty(n, dtype="U64")
        channel_std = np.empty((n, 3))
        chunk = handle["images"].chunks[0] if handle["images"].chunks else 128
        for start in range(0, n, chunk):
            for offset, image in enumerate(handle["images"][start:start+chunk]):
                i = start + offset
                digests[i] = hashlib.sha256(image.tobytes()).hexdigest()
                channel_std[i] = image.reshape(-1, 3).std(0)
    groups, conflicting, conflicts, pairs = build_groups(labels, digests, ra, dec)
    blank_candidates = np.flatnonzero(channel_std.max(1) < 2)
    # Only these frames have repository-backed visual confirmation. New candidates
    # require inspection; never silently broaden P-BLANK to new data.
    confirmed = np.array([11492, 12659, 15422], dtype=np.int64)
    if not set(blank_candidates) <= set(confirmed):
        raise ValueError(f"Unreviewed blank candidates: {blank_candidates.tolist()}")
    blanks = np.intersect1d(blank_candidates, confirmed)
    excluded = np.union1d(conflicting, blanks)
    splits = generate_splits(labels, groups, excluded)
    integrity = validate_splits(splits, excluded, groups, n)
    old = {name: sha256(folder / f"{name}_idx.npy") for name in (*splits, "excluded") if (folder / f"{name}_idx.npy").exists()}
    names = [f"{name}_idx.npy" for name in (*splits, "excluded")] + ["split_report.json", "group_ids.npy", "pixel_scale.npy", "image_digests.csv"]
    archive_files(folder, names)
    for name, values in {**splits, "excluded": excluded}.items():
        np.save(folder / f"{name}_idx.npy", values)
    np.save(folder / "group_ids.npy", groups)
    np.save(folder / "pixel_scale.npy", scale)
    (folder / "image_digests.csv").write_text("index,sha256,group_id\n" + "".join(f"{i},{digest},{groups[i]}\n" for i, digest in enumerate(digests)), encoding="utf-8")
    new = {name: sha256(folder / f"{name}_idx.npy") for name in (*splits, "excluded")}
    levels = np.unique(scale[np.isfinite(scale)])
    report = {
        "schema_version": 2, "status": "audited_unfrozen", "seed": 42,
        "configuration": {"algorithm": "nested StratifiedGroupKFold", "outer_folds": 10, "inner_folds": 9,
                          "selected_fold": 0, "shuffle": True, "tolerance_arcsec": 3.0,
                          "group_id": "minimum index of transitive exact-pixel/sky component", "pixel_scale_stratification": False},
        "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        "n_images": n, "excluded_images": len(excluded),
        "conflicting_duplicate_groups": len(conflicts), "conflicting_duplicate_images": len(conflicting),
        "conflicting_images_per_class": np.bincount(labels[conflicting], minlength=10).tolist(),
        "conflicting_groups_per_class": [sum(c in labels[g] for g in conflicts) for c in range(10)],
        "blank_candidates": blank_candidates.tolist(), "visually_confirmed_blanks": blanks.tolist(),
        "blank_confirmation_source": "notebooks/01_dataset_exploration.ipynb cell 7d / findings",
        "sky_pairs": len(pairs), "integrity": integrity,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": {k: np.bincount(labels[v], minlength=10).tolist() for k, v in {**splits, "excluded": excluded}.items()},
        "group_counts": {k: len(np.unique(groups[v])) for k, v in splits.items()},
        "pixel_scale_share": {k: {str(s): float(np.mean(np.isclose(scale[v], s, rtol=0, atol=1e-9))) for s in levels} for k, v in splits.items()},
        "pixel_scale_class_counts_full_dataset": {str(s): np.bincount(labels[np.isclose(scale, s, rtol=0, atol=1e-9)], minlength=10).tolist() for s in levels},
        "input_sha256": {"raw_hdf5": sha256(raw), "labels": sha256(root / "data/processed/galaxy10_labels.npy"), "data_config": sha256(root / "configs/data.yaml")},
        "output_sha256": {name: sha256(folder / name) for name in names if name != "split_report.json"},
        "split_fingerprints": new,
    }
    write_json(folder / "split_report.json", report)
    # Comparison belongs outside the deterministic report: regenerating identical
    # inputs must produce a byte-identical report.
    audit = {"previous_split_fingerprints": old, "current_split_fingerprints": new,
             "changed": old != new, "existing_results": "preliminary; compare only matching data, splits and normalization"}
    from galaxy10.audit import unique_results
    out = unique_results(root, "split-audit")
    write_json(out / "comparison.json", audit)
    print(audit)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    prepare(parser.parse_args().root)
