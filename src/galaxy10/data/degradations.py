"""Display-space perturbations; evaluation RNGs never depend on loader order."""
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F


def sample_seed(seed, index, kind, severity):
    key = f"{int(seed)}:{int(index)}:{kind}:{float(severity).hex()}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def degrade(image, kind, severity, *, seed=42, index=0):
    """Input/output CHW float [0,1]. Noise is independent per pixel/channel."""
    value = float(severity)
    if kind == "clean":
        return image.clone()
    if not math.isfinite(value):
        raise ValueError("Severity must be finite")
    if kind == "gaussian_noise":
        if value < 0:
            raise ValueError("Noise sigma must be nonnegative")
        generator = torch.Generator(device="cpu").manual_seed(sample_seed(seed, index, kind, value))
        image = image + value * torch.randn(image.shape, generator=generator, dtype=image.dtype).to(image.device)
    elif kind == "gaussian_blur":
        if value < 0:
            raise ValueError("Blur sigma must be nonnegative")
        if value > 0:
            radius = math.ceil(3 * value)
            axis = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
            kernel = torch.exp(-axis.square() / (2 * value * value))
            kernel /= kernel.sum()
            channels = image.shape[0]
            x = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
            kernel2d = (kernel[:, None] * kernel[None, :]).expand(channels, 1, -1, -1)
            image = F.conv2d(x, kernel2d, groups=channels)[0]
    elif kind == "resolution":
        if value < 1 or value != int(value) or image.shape[1] % int(value) or image.shape[2] % int(value):
            raise ValueError("Resolution factor must be a positive integer dividing the image dimensions")
        import cv2
        array = image.detach().cpu().permute(1, 2, 0).numpy()
        height, width = array.shape[:2]
        small = cv2.resize(array, (width // int(value), height // int(value)), interpolation=cv2.INTER_AREA)
        array = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).to(image)
    elif kind == "contrast":
        if not 0 <= value <= 1:
            raise ValueError("Contrast must be in [0,1]")
        mean = image.mean()
        image = mean + value * (image - mean)
    else:
        raise ValueError(f"Unknown degradation: {kind}")
    return image.clamp(0, 1)


class TargetedAugmentation:
    """Separate RNG preserves the clean condition's geometric RNG draws."""
    def __init__(self, seed):
        self.seed = seed
        self.generator = None

    def __call__(self, image):
        if self.generator is None:
            worker = torch.utils.data.get_worker_info()
            worker_seed = worker.seed if worker else self.seed
            self.generator = torch.Generator().manual_seed(sample_seed(worker_seed, 0, "augmentation", 0))
        rng = self.generator
        if torch.rand((), generator=rng).item() >= 0.5:
            return image
        kind = "gaussian_noise" if torch.randint(2, (), generator=rng).item() == 0 else "gaussian_blur"
        severity = torch.rand((), generator=rng).item() * (0.10 if kind == "gaussian_noise" else 2.0)
        draw_seed = torch.randint(2**31, (), generator=rng).item()
        return degrade(image, kind, severity, seed=draw_seed)
