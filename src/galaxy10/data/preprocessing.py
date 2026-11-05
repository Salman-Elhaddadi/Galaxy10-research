from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = ROOT / "data" / "processed" / "train_normalization.npz"


def load_normalization_stats(path=None) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path) if path else STATS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found: {STATS_PATH}. "
            "Run the normalization-statistics script first."
        )

    with np.load(path, allow_pickle=False) as stats:
        mean = stats["mean"]
        std = stats["std"]

    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Expected three RGB mean and standard-deviation values")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Normalization statistics must be finite")
    if (std <= 0).any():
        raise ValueError("Normalization standard deviations must be positive")

    return (
        torch.tensor(mean, dtype=torch.float32).view(3, 1, 1),
        torch.tensor(std, dtype=torch.float32).view(3, 1, 1),
    )


def normalize_image(
    image: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (image - mean) / std
