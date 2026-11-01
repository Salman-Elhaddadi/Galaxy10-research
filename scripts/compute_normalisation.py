from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGES_PATH = ROOT / "data" / "processed" / "galaxy10_128x128_uint8.npy"
TRAIN_PATH = ROOT / "splits" / "train_idx.npy"
OUTPUT_PATH = ROOT / "data" / "processed" / "train_normalization.npz"

BATCH_SIZE = 128

images = np.load(IMAGES_PATH, mmap_mode="r", allow_pickle=False)
train_indices = np.load(TRAIN_PATH, allow_pickle=False)

channel_sum = np.zeros(3, dtype=np.float64)
channel_sum_sq = np.zeros(3, dtype=np.float64)
pixel_count = 0

for start in range(0, len(train_indices), BATCH_SIZE):
    indices = train_indices[start : start + BATCH_SIZE]
    batch = images[indices].astype(np.float64) / 255.0

    channel_sum += batch.sum(axis=(0, 1, 2))
    channel_sum_sq += np.square(batch).sum(axis=(0, 1, 2))
    pixel_count += batch.shape[0] * batch.shape[1] * batch.shape[2]

mean = channel_sum / pixel_count
variance = channel_sum_sq / pixel_count - np.square(mean)
std = np.sqrt(np.maximum(variance, 0))

np.savez(OUTPUT_PATH, mean=mean, std=std)

print("Training-split channel mean:", mean.tolist())
print("Training-split channel std:", std.tolist())
print(f"Saved statistics to: {OUTPUT_PATH}")