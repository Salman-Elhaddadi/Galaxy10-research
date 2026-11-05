"""Training-only intrinsic border MAD and one-pass CPU Model A benchmark."""
import argparse
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from galaxy10.audit import sha256, unique_results, write_json
from galaxy10.evaluate import environment
from galaxy10.model.CNN import ModelA
from galaxy10.model.train import seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root
    seed_everything(42)
    torch.set_num_threads(1)
    path = root / "data/processed/galaxy10_128x128_uint8.npy"
    images = np.load(path, mmap_mode="r", allow_pickle=False)
    indices = np.load(root / "splits/train_idx.npy", allow_pickle=False)
    border = np.ones((128, 128), dtype=bool)
    border[8:-8, 8:-8] = False  # Same fractional border as 16px at 256.
    noise = []
    for start in range(0, len(indices), 128):
        batch = images[indices[start:start+128]].astype(np.float32) / 255
        pixels = batch[:, border, :]
        median = np.median(pixels, axis=1, keepdims=True)
        noise.extend(1.4826 * np.median(np.abs(pixels - median), axis=1))
    noise = np.asarray(noise)
    medians = np.median(noise, axis=0)
    model = ModelA()
    x = torch.zeros(1, 3, 128, 128)
    macs = []
    def count_macs(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            macs.append(output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size).item())
        elif isinstance(module, torch.nn.Linear):
            macs.append(output.numel() * module.in_features)
    hooks = [m.register_forward_hook(count_macs) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    start = time.perf_counter()
    logits = model(x)
    forward = time.perf_counter() - start
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    start = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    report = {"schema_version": 1, "scope": "training images only; no test access", "n_train": len(indices),
              "images_sha256": sha256(path), "train_indices_sha256": sha256(root / "splits/train_idx.npy"),
              "noise": {"method": "per-channel 1.4826 * border MAD; 8px border at 128; RGB [0,1]",
                        "percentiles_1_50_99_per_channel": np.percentile(noise, [1, 50, 99], axis=0).tolist(),
                        "added_sigma_to_median_intrinsic_ratio": {str(s): (s / medians).tolist() for s in (0, .02, .05, .10, .15)}},
              "benchmark": {"batch_size": 1, "threads": 1, "warmup": 0, "forward_seconds": forward,
                            "backward_seconds": backward, "input": "synthetic zeros 1x3x128x128", "loss": loss.item(),
                            "parameters": sum(p.numel() for p in model.parameters()), "conv_linear_MACs": sum(macs),
                            "single_batch_overfit": "not performed by this command"}, **environment("cpu")}
    output = unique_results(root, "preparation-audit")
    write_json(output / "audit.json", report)
    print(output)


if __name__ == "__main__":
    main()
