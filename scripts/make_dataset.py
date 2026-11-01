from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Galaxy10_DECals.h5"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128

with h5py.File(RAW, "r") as f:
    images = f["images"]
    labels = f["ans"][:].astype(np.int64)
    n = images.shape[0]

    # Align reads with the HDF5 image chunk size when available.
    batch_size = images.chunks[0] if images.chunks else 128

    output = np.lib.format.open_memmap(
        PROCESSED / "galaxy10_128x128_uint8.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n, IMG_SIZE, IMG_SIZE, 3),
    )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images[start:end]

        for j, image in enumerate(batch):
            output[start + j] = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_AREA,
            )

        print(f"Processed {end}/{n} images")

    output.flush()

np.save(PROCESSED / "galaxy10_labels.npy", labels)
print("Saved resized images and labels.")