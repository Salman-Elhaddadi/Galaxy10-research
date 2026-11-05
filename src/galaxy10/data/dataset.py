from pathlib import Path
from galaxy10.data.preprocessing import load_normalization_stats, normalize_image
import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"


class Galaxy10Dataset(Dataset):
    def __init__(self, indices: np.ndarray, transform=None, *, root=None, degradation=None, degradation_seed=42):
        processed = Path(root) / "data/processed" if root is not None else PROCESSED
        self.mean, self.std = load_normalization_stats(processed / "train_normalization.npz")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.degradation = degradation
        self.degradation_seed = degradation_seed

        self.images = np.load(
            processed / "galaxy10_128x128_uint8.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        self.labels = np.load(
            processed / "galaxy10_labels.npy",
            allow_pickle=False,
        )

        if self.images.shape[0] != len(self.labels):
            raise ValueError("Image and label counts do not match")
        if self.images.shape[1:] != (128, 128, 3):
            raise ValueError(f"Unexpected image shape: {self.images.shape}")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.labels)):
            raise IndexError("Dataset indices are out of range")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[item]

        # Copy the read-only, memory-mapped image before creating a tensor.
        image = torch.from_numpy(
            self.images[index].transpose(2, 0, 1).copy()
        ).to(torch.float32).div_(255.0)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        if self.degradation is not None:
            from galaxy10.data.degradations import degrade
            image = degrade(image, *self.degradation, seed=self.degradation_seed, index=int(index))

        image = normalize_image(image.clamp(0, 1), self.mean, self.std)
        return image, label

