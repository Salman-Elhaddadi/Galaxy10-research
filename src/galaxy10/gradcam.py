"""Deterministic, preselected Grad-CAM views from saved evaluation predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from galaxy10.audit import ROOT, authorize_split, data_fingerprints, sha256, unique_results, write_json
from galaxy10.data.degradations import degrade
from galaxy10.data.preprocessing import load_normalization_stats, normalize_image
from galaxy10.evaluate import environment, load_checkpoint
from galaxy10.model.CNN import ModelA
from galaxy10.model.train import seed_everything


def select_examples(indices, labels, predictions, seed=42, per_class=4):
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        for correct in (True, False):
            candidates = np.sort(np.asarray(indices)[(labels == label) & ((predictions == labels) == correct)])
            for index in np.sort(rng.choice(candidates, min(per_class, len(candidates)), replace=False)):
                selected.append({"index": int(index), "label": label, "clean_correct": correct})
    return selected


def gradcam(model, image):
    activation = []
    # Hook the final ReLU output: BN/ReLU is part of the last convolution block,
    # immediately before AdaptiveAvgPool, with shape 256x16x16 at 128 input.
    handle = model.features[14].register_forward_hook(lambda module, args, output: activation.append(output))
    try:
        model.eval()
        logits = model(image[None].requires_grad_(True))
        target = int(logits.argmax(1).item())
        gradient = torch.autograd.grad(logits[0, target], activation[0])[0]
        weights = gradient.mean((2, 3), keepdim=True)
        heatmap = (weights * activation[0]).sum(1, keepdim=True).relu()
        heatmap = F.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        heatmap = heatmap / heatmap.max().clamp_min(1e-12)
        return heatmap.detach().cpu().numpy(), target
    finally:
        handle.remove()


def run_gradcam(evaluation_dir, *, root=ROOT, frozen_test=False, freeze_manifest=None, per_class=4, seed=42, device="cpu"):
    root, evaluation_dir = Path(root).resolve(), Path(evaluation_dir).resolve()
    config = json.loads((evaluation_dir / "config.json").read_text())
    split = config["split"]
    authorize_split(root, split, frozen_test, freeze_manifest)
    if json.loads((evaluation_dir / "summary.json").read_text())["status"] != "completed":
        raise ValueError("Requires a completed evaluation")
    if config["model"] != "a" or config["training_seed"] != 0:
        raise ValueError("Protocol Grad-CAM requires Model A training seed 0")
    if config["data_fingerprints"] != data_fingerprints(root, include_test=split == "test"):
        raise ValueError("Evaluation data changed")
    checkpoint_path = Path(config["checkpoint"])
    if sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    seed_everything(seed)
    model, _ = load_checkpoint(checkpoint_path, device)
    with np.load(evaluation_dir / "clean-0/predictions.npz", allow_pickle=False) as saved:
        selected = select_examples(saved["indices"], saved["labels"], saved["probabilities"].argmax(1), seed, per_class)
        if not np.array_equal(saved["indices"], np.load(root / f"splits/{split}_idx.npy", allow_pickle=False)):
            raise ValueError("Prediction indices do not match selected split")
    import yaml
    degradation_config = yaml.safe_load((root / "configs/degradations.yaml").read_text())
    if sha256(root / "configs/degradations.yaml") != config["degradation_config_sha256"]:
        raise ValueError("Degradation configuration changed after evaluation")
    output = unique_results(root, f"gradcam-{split}")
    selection = {"schema_version": 1, "model": "a", "training_seed": 0, "condition": config["condition"], "split": split, "seed": seed, "per_class_per_outcome": per_class,
                 "selection": "seeded sampling within true class and clean correct/incorrect outcomes; sorted candidates",
                 "target_layer": "features[14] final convolution block ReLU output 256x16x16",
                 "target_class": "each view's predicted class", "randomization": "all weights freshly initialized; BN reset; eval mode",
                 "randomization_seed": seed, "mid_levels": degradation_config["gradcam_mid"],
                 "evaluation": str(evaluation_dir), "checkpoint_sha256": config["checkpoint_sha256"],
                 "prediction_sha256": sha256(evaluation_dir / "clean-0/predictions.npz"), "selected": selected,
                 "qualitative_only": True, **environment(device)}
    # Persist before computing or viewing a single heatmap.
    write_json(output / "selection.json", selection)
    seed_everything(seed)
    randomized = ModelA().to(device).eval()
    mean, std = load_normalization_stats(root / "data/processed/train_normalization.npz")
    images = np.load(root / "data/processed/galaxy10_128x128_uint8.npy", mmap_mode="r", allow_pickle=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = []
    for item in selected:
        i = item["index"]
        clean = torch.from_numpy(images[i].transpose(2, 0, 1).copy()).float() / 255
        views = [("clean", 0), *degradation_config["gradcam_mid"].items()]
        fig, axes = plt.subplots(3, len(views), figsize=(3 * len(views), 8), squeeze=False)
        for col, (kind, severity) in enumerate(views):
            view = degrade(clean, kind, severity, seed=degradation_config["seed"], index=i)
            normalized = normalize_image(view, mean, std).to(device)
            rgb = view.permute(1, 2, 0).numpy()
            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"{kind} {severity}")
            for row, (label, network) in enumerate((("trained", model), ("randomized", randomized)), 1):
                heatmap, target = gradcam(network, normalized.clone())
                np.save(output / f"index{i}-{kind}-{label}.npy", heatmap)
                axes[row, col].imshow(rgb)
                axes[row, col].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[row, col].set_title(f"{label}: predicted {target}")
                records.append({"index": i, "kind": kind, "severity": severity, "weights": label, "target_class": target})
            for row in range(3):
                axes[row, col].axis("off")
        fig.suptitle(f"Index {i}, class {item['label']}; display-space perturbations; qualitative Grad-CAM")
        fig.tight_layout()
        fig.savefig(output / f"index{i}.png")
        plt.close(fig)
    write_json(output / "views.json", records)
    write_json(output / "summary.json", {"status": "completed", "selected_images": len(selected), "views": len(records)})
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.per_class < 1 or args.per_class > 4 or args.seed != 42:
        parser.error("Protocol selection requires seed 42 and 1..4 examples per class/outcome")
    run_gradcam(**vars(args))


if __name__ == "__main__":
    main()
