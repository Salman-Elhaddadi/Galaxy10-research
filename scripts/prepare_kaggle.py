"""Prepare a writable Kaggle checkout using committed audited splits, without runs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from galaxy10.audit import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source", type=Path, help="Defaults to the sole Galaxy10_DECals.h5 under /kaggle/input")
    args = parser.parse_args(argv)
    raw = ROOT / "data/raw/Galaxy10_DECals.h5"
    if not raw.exists():
        candidates = [args.raw_source] if args.raw_source else list(Path("/kaggle/input").rglob("Galaxy10_DECals.h5"))
        if len(candidates) != 1:
            parser.error("Expected exactly one raw file; specify --raw-source explicitly")
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("xb") as dest, candidates[0].open("rb") as source:
            shutil.copyfileobj(source, dest)
    prepared_config = json.loads((ROOT / "configs/prepared_data.json").read_text())
    expected = prepared_config["file_sha256"]
    if sha256(raw) != expected["data/raw/Galaxy10_DECals.h5"]:
        raise ValueError("Raw dataset fingerprint differs from the audited dataset")
    processed = ROOT / "data/processed"
    paths = [processed / "galaxy10_128x128_uint8.npy", processed / "galaxy10_labels.npy"]
    if not any(p.exists() for p in paths):
        subprocess.run([sys.executable, str(ROOT / "scripts/make_dataset.py")], check=True)
    if not all(p.exists() for p in paths):
        raise ValueError("Incomplete cache; inspect it before rebuilding")
    for path in paths:
        key = path.relative_to(ROOT).as_posix()
        if sha256(path) != expected[key]:
            raise ValueError(f"Cache differs from audited preparation: {key}")
    # This uses committed splits; do not regenerate splits with Kaggle's sklearn.
    subprocess.run([sys.executable, str(ROOT / "scripts/check_spatial_leakage.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/compute_normalisation.py")], check=True)
    # NPZ is a ZIP container, so byte serialization can vary across Python/NumPy
    # versions even when its arrays are identical. Validate the scientific values
    # and bound provenance rather than requiring a cross-platform container hash.
    with np.load(processed / "train_normalization.npz", allow_pickle=False) as normalization:
        if set(normalization.files) != {"mean", "std"}:
            raise ValueError("Unexpected normalization archive schema")
        mean = normalization["mean"]
        std = normalization["std"]
    expected_mean = np.asarray(prepared_config["normalization_mean"], dtype=np.float64)
    expected_std = np.asarray(prepared_config["normalization_std"], dtype=np.float64)
    mean_matches = np.allclose(mean, expected_mean, rtol=0, atol=1e-12)
    std_matches = np.allclose(std, expected_std, rtol=0, atol=1e-12)
    if mean.shape != (3,) or std.shape != (3,) or not mean_matches or not std_matches:
        raise ValueError("Normalization numerics differ from the audited preparation")
    provenance = json.loads((processed / "normalization.json").read_text())
    train_path = ROOT / "splits/train_idx.npy"
    expected_n_train = len(np.load(train_path, allow_pickle=False))
    provenance_matches = (
        provenance.get("images_sha256") == expected["data/processed/galaxy10_128x128_uint8.npy"]
        and provenance.get("train_indices_sha256") == sha256(train_path)
        and provenance.get("n_train") == expected_n_train
        and provenance.get("test_access") is False
    )
    if not provenance_matches:
        raise ValueError("Normalization provenance differs from the audited preparation")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_data.py")], check=True)
    print("Prepared corrected development data; no training or test evaluation performed.")


if __name__ == "__main__":
    main()
