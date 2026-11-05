"""Check development batches; test pixels require explicit frozen authorization."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from torch.utils.data import DataLoader
from galaxy10.audit import authorize_split
from galaxy10.data.dataset import Galaxy10Dataset
from galaxy10.data.splits import load_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    args = parser.parse_args()
    if args.include_test:
        authorize_split(ROOT, "test", args.frozen_test, args.freeze_manifest)
    splits = load_splits(ROOT)
    for name in ("train", "val", "test") if args.include_test else ("train", "val"):
        dataset = Galaxy10Dataset(splits[name])
        images, labels = next(iter(DataLoader(dataset, batch_size=8)))
        assert images.shape == (8, 3, 128, 128) and images.dtype == torch.float32
        assert labels.shape == (8,) and labels.dtype == torch.long
        print(f"{name}: {len(dataset)} samples; batch {tuple(images.shape)}")
