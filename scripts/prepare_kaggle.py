"""Prepare a writable Kaggle checkout using committed audited splits, without runs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
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
    expected = json.loads((ROOT / "configs/prepared_data.json").read_text())["file_sha256"]
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
    if sha256(processed / "train_normalization.npz") != expected["data/processed/train_normalization.npz"]:
        raise ValueError("Normalization serialization/numerics differ; audit and record before comparing runs")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_data.py")], check=True)
    print("Prepared corrected development data; no training or test evaluation performed.")


if __name__ == "__main__":
    main()
