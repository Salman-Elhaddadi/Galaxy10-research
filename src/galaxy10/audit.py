"""Shared artifact and freeze integrity helpers (no test-image reads)."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def unique_results(root, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(root) / "results" / f"{prefix}-{stamp}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def archive_files(folder, names):
    """Content-addressed snapshot; leave completed results/checkpoints untouched."""
    files = [Path(folder) / name for name in names if (Path(folder) / name).exists()]
    if not files:
        return None
    key = hashlib.sha256(json.dumps({p.name: sha256(p) for p in files}, sort_keys=True).encode()).hexdigest()
    dest = Path(folder) / "history" / key
    dest.mkdir(parents=True, exist_ok=True)
    for path in files:
        target = dest / path.name
        if not target.exists():
            shutil.copy2(path, target)
        elif sha256(target) != sha256(path):
            raise ValueError("History snapshot collision")
    return str(dest)


def data_fingerprints(root, *, include_test=False):
    root = Path(root)
    names = ["data/processed/galaxy10_128x128_uint8.npy", "data/processed/galaxy10_labels.npy",
             "data/processed/train_normalization.npz", "splits/train_idx.npy", "splits/val_idx.npy"]
    if include_test:
        names += ["splits/test_idx.npy", "splits/excluded_idx.npy", "splits/group_ids.npy", "splits/pixel_scale.npy"]
    return {name: sha256(root / name) for name in names}


def freeze_fingerprints(root, exclude=()):
    root = Path(root)
    excluded = {Path(p).resolve() for p in exclude}
    hashes = data_fingerprints(root, include_test=True)
    for folder, pattern in [("src", "*.py"), ("configs", "*.yaml"), ("configs", "*.json")]:
        hashes.update({str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sorted((root / folder).rglob(pattern)) if p.resolve() not in excluded})
    hashes["docs/protocol.md"] = sha256(root / "docs/protocol.md")
    hashes["splits/split_report.json"] = sha256(root / "splits/split_report.json")
    return hashes


def authorize_split(root, split, frozen_test=False, freeze_manifest=None):
    if split not in ("val", "test"):
        raise ValueError("Explicitly select val or test")
    if split != "test":
        return None
    if not frozen_test:
        raise ValueError("Test access requires --frozen-test and a matching --freeze-manifest")
    if freeze_manifest is None:
        raise ValueError("Test access requires --freeze-manifest")
    manifest = json.loads(Path(freeze_manifest).read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or not manifest.get("checks") or not all(v is True for v in manifest["checks"].values()):
        raise ValueError("Protocol and splits remain unfrozen")
    required = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "single_batch_sanity", "deliberate_test_decision"}
    if not required <= manifest["checks"].keys() or not all(manifest.get("evidence", {}).get(k) for k in required):
        raise ValueError("Freeze manifest requires evidence for every prerequisite")
    if manifest.get("fingerprints") != freeze_fingerprints(root, exclude=[freeze_manifest]):
        raise ValueError("Frozen protocol/data/source fingerprints do not match")
    return manifest
