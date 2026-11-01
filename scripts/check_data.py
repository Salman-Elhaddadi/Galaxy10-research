import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from galaxy10.data.dataset import Galaxy10Dataset
from galaxy10.data.splits import load_splits


splits = load_splits()

for name, indices in splits.items():
    dataset = Galaxy10Dataset(indices)
    images, labels = next(iter(DataLoader(dataset, batch_size=8)))

    assert images.shape == (8, 3, 128, 128), images.shape
    assert images.dtype == torch.float32, images.dtype
    assert labels.shape == (8,), labels.shape
    assert labels.dtype == torch.long, labels.dtype

    print(f"{name}: {len(dataset)} samples; batch {tuple(images.shape)}")