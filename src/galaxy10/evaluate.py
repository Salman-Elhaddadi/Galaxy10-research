"""Saved-best-checkpoint evaluation; explicit validation or protected frozen test."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from galaxy10.audit import ROOT, authorize_split, data_fingerprints, sha256, unique_results, write_json
from galaxy10.data.dataset import Galaxy10Dataset
from galaxy10.metrics import bootstrap, classification_metrics, pixel_scale_metrics
from galaxy10.model.CNN import ModelA
from galaxy10.model.residual import ModelB
from galaxy10.model.train import seed_everything


def environment(device):
    versions = {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda}
    for name in ("scipy", "scikit-learn", "opencv-python", "h5py", "PyYAML", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions, "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count(), "device": str(device), "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}}


def load_checkpoint(path, device="cpu"):
    """Require this repository's saved best format; reject raw/final state dicts."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "epoch", "selection_metric", "validation", "config"}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError("Expected saved best checkpoint with config and clean validation selection metadata")
    if checkpoint["selection_metric"] != "clean_validation_macro_f1":
        raise ValueError("Checkpoint selection metric is incompatible")
    config = checkpoint["config"]
    model_name = config.get("model")
    if model_name not in ("a", "b"):
        raise ValueError("Checkpoint must identify architecture a or b")
    model = {"a": ModelA, "b": ModelB}[model_name]()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if config.get("parameter_count", sum(p.numel() for p in model.parameters())) != sum(p.numel() for p in model.parameters()):
        raise ValueError("Checkpoint parameter count differs from architecture")
    return model.to(device).eval(), checkpoint


def checkpoint_compatibility(checkpoint, fingerprints):
    recorded = checkpoint["config"].get("provenance", {}).get("file_sha256", {})
    recorded = {key.replace("\\", "/"): value for key, value in recorded.items()}
    expected = {key: value for key, value in fingerprints.items() if key.startswith("data/") or key in ("splits/train_idx.npy", "splits/val_idx.npy")}
    return [key for key, value in expected.items() if recorded.get(key) != value]


def predict(model, dataset, device, batch_size=32):
    probabilities, labels, losses = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, target in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = model(images.to(device))
            logp = logits.log_softmax(1).cpu()
            probabilities.append(logp.exp().double().numpy())
            labels.append(target.numpy())
            losses.append((-logp[torch.arange(len(target)), target]).double().numpy())
    if not labels:
        raise ValueError("Empty evaluation split")
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(losses)


def save_predictions(folder, indices, labels, probabilities, nll, scales):
    np.savez_compressed(folder / "predictions.npz", indices=indices, labels=labels,
                        probabilities=probabilities, nll=nll, pixel_scale=scales)
    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "label", "prediction", "confidence", "nll", "pixel_scale", *[f"p_{i}" for i in range(10)]])
        for i, label, p, loss, scale in zip(indices, labels, probabilities, nll, scales):
            writer.writerow([int(i), int(label), int(p.argmax()), float(p.max()), float(loss), float(scale), *p.tolist()])


def exposure_group(kind, severity):
    if kind == "clean" or (kind in ("gaussian_noise", "gaussian_blur") and severity == 0) or (kind in ("resolution", "contrast") and severity == 1):
        return "clean"
    if kind in ("resolution", "contrast"):
        return "unseen_type"
    maximum = .10 if kind == "gaussian_noise" else 2.0
    return "seen_type_seen_severity" if severity <= maximum else "seen_type_extrapolated_severity"


def plots(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in sorted({r["kind"] for r in records} - {"clean"}):
        values = [r for r in records if r["kind"] == kind]
        x = [r["severity"] for r in values]
        fig, ax = plt.subplots()
        ax.plot(x, [r["metrics"]["macro_f1"] for r in values], "o-")
        ax.axhline(records[0]["metrics"]["macro_f1"], ls="--", label="clean")
        if kind == "contrast":
            ax.invert_xaxis()
        ax.set(xlabel=f"{kind} severity (display-space perturbation)", ylabel="Macro-F1", ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{kind}_macro_f1.png")
        plt.close(fig)
        fig, ax = plt.subplots()
        im = ax.imshow(np.array([r["metrics"]["per_class_f1"] for r in values]).T, vmin=0, vmax=1, aspect="auto")
        ax.set(xticks=np.arange(len(x)), xticklabels=x, yticks=np.arange(10), xlabel=f"{kind} severity (display-space)", ylabel="Class")
        fig.colorbar(im, ax=ax, label="F1")
        fig.tight_layout()
        fig.savefig(output / f"{kind}_per_class_f1.png")
        plt.close(fig)
    reliability = [records[0]] + [next(r for r in reversed(records) if r["kind"] == kind) for kind in sorted({r["kind"] for r in records} - {"clean"})]
    for record in reliability:
        bins = [b for b in record["metrics"]["reliability_bins"] if b["count"]]
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
        ax.set(xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1), title=f"{record['kind']} {record['severity']} (display-space)")
        fig.tight_layout()
        fig.savefig(output / f"reliability_{record['name']}.png")
        plt.close(fig)


def evaluate(root, checkpoint_path, split, *, frozen_test=False, freeze_manifest=None,
             degradations=False, allow_preliminary=False, device="cpu", batch_size=32, seed=42):
    root, checkpoint_path = Path(root).resolve(), Path(checkpoint_path).resolve()
    manifest = authorize_split(root, split, frozen_test, freeze_manifest)  # Before checkpoint, indices or pixels.
    seed_everything(seed)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    fingerprints = data_fingerprints(root, include_test=split == "test")
    mismatches = checkpoint_compatibility(checkpoint, fingerprints)
    if mismatches and (split == "test" or not allow_preliminary):
        raise ValueError(f"Checkpoint/data fingerprints missing or incompatible: {mismatches}; validation-only --allow-preliminary permits diagnostics")
    if split == "test" and checkpoint["config"].get("formal_results_eligible", False) and checkpoint["config"].get("protocol_manifest_sha256") != sha256(freeze_manifest):
        raise ValueError("Formal checkpoint was trained against a different frozen manifest")
    config_path = root / "configs/degradations.yaml"
    import yaml
    degradation_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if seed != degradation_config["seed"]:
        raise ValueError("Evaluation seed must match frozen degradation configuration")
    conditions = [("clean", 0)]
    if degradations:
        conditions += [(kind, severity) for kind, levels in degradation_config["levels"].items() for severity in levels]
    output = unique_results(root, f"evaluation-{split}")
    prepared = root / "configs/prepared_data.json"
    noise_reference = json.loads(prepared.read_text()).get("intrinsic_noise_reference") if prepared.exists() else None
    config = {"schema_version": 1, "workflow": "evaluation", "split": split,
              "status": "frozen_test" if split == "test" else "validation_diagnostic",
              "formal_results_eligible": split == "test" and checkpoint["config"]["model"] == "a" and not mismatches and checkpoint["config"].get("formal_results_eligible", False),
              "model": checkpoint["config"]["model"], "condition": checkpoint["config"].get("condition", "clean"),
              "training_seed": checkpoint["config"]["seed"], "seed": seed, "batch_size": batch_size,
              "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
              "selected_epoch": checkpoint["epoch"], "checkpoint_config": checkpoint["config"],
              "data_fingerprints": fingerprints, "preliminary_mismatches": mismatches,
              "freeze_manifest": manifest, "degradations": degradation_config if degradations else None,
              "degradation_config_sha256": sha256(config_path),
              "intrinsic_noise_reference": noise_reference,
              "evaluation_source_sha256": {str(p.relative_to(Path(__file__).parent)): sha256(p) for p in sorted(Path(__file__).parent.rglob("*.py"))},
              "protocol_sha256": sha256(root / "docs/protocol.md") if (root / "docs/protocol.md").exists() else None,
              "bootstrap": {"resamples": 1000, "confidence": 0.95, "seed": seed, "unit": "image"},
              **environment(device)}
    write_json(output / "config.json", config)
    write_json(output / "summary.json", {"status": "running"})
    # One frozen metrics pass per saved checkpoint and frozen data/configuration.
    # A failed reservation is retained for audit; resolve any retry deliberately.
    try:
        if split == "test":
            import hashlib
            key = hashlib.sha256((config["checkpoint_sha256"] + json.dumps(manifest, sort_keys=True)).encode()).hexdigest()
            ledger = root / "results/test_access"
            ledger.mkdir(parents=True, exist_ok=True)
            with (ledger / f"{key}.json").open("x", encoding="utf-8") as handle:
                json.dump({"output": str(output), "checkpoint_sha256": config["checkpoint_sha256"]}, handle)
        indices = np.load(root / f"splits/{split}_idx.npy", allow_pickle=False)
        # Validate index/group/exclusion metadata before reading any selected pixels.
        from galaxy10.data.splits import load_splits
        audited = load_splits(root)
        if not np.array_equal(indices, audited[split]):
            raise ValueError("Split changed during evaluation")
        scales = np.load(root / "splits/pixel_scale.npy", allow_pickle=False)[indices]
        records = []
        clean_probabilities = None
        clean_nll = None
        for kind, severity in conditions:
            name = f"{kind}-{float(severity):g}"
            folder = output / name
            folder.mkdir()
            dataset = Galaxy10Dataset(indices, root=root, degradation=(kind, severity), degradation_seed=seed)
            labels, p, nll = predict(model, dataset, device, batch_size)
            metrics = classification_metrics(labels, p, nll)
            metrics["by_pixel_scale"] = pixel_scale_metrics(labels, p, scales, nll)
            ci = bootstrap(labels, p, seed=seed, nll=nll)
            if kind == "clean":
                clean_probabilities = p
                clean_nll = nll
            else:
                ci["paired_drop_from_clean"] = bootstrap(labels, clean_probabilities, seed=seed, nll=clean_nll, reference=p, reference_nll=nll)
                ci["paired_drop_from_clean"]["difference"] = "clean_minus_degraded"
            save_predictions(folder, indices, labels, p, nll, scales)
            write_json(folder / "metrics.json", metrics)
            write_json(folder / "confidence_intervals.json", ci)
            record = {"name": name, "kind": kind, "severity": severity, "augmentation_exposure_group": exposure_group(kind, severity), "metrics": metrics, "confidence_intervals": ci}
            if kind == "gaussian_noise" and noise_reference:
                record["added_sigma_over_training_median_intrinsic_rgb"] = (severity / np.array(noise_reference["median_sigma_rgb"])).tolist()
            records.append(record)
        write_json(output / "metrics.json", {"schema_version": 1, "model": config["model"], "condition": config["condition"], "training_seed": config["training_seed"], "split": split, "records": records})
        with (output / "severity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["kind", "severity", "augmentation_exposure_group", "macro_f1", "accuracy", "ece_15", "brier_score", "nll", *[f"f1_class_{i}" for i in range(10)]])
            for r in records:
                m = r["metrics"]
                writer.writerow([r["kind"], r["severity"], r["augmentation_exposure_group"], *[m[k] for k in ("macro_f1", "accuracy", "ece_15", "brier_score", "nll")], *m["per_class_f1"]])
        plots(output, records)
        write_json(output / "summary.json", {"status": "completed", "split": split, "records": len(records), "test_access": split == "test"})
    except BaseException as error:
        write_json(output / "summary.json", {"status": "failed", "error": str(error)})
        raise
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--frozen-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--degradations", action="store_true", help="Clean plus all frozen degradation levels")
    parser.add_argument("--allow-preliminary", action="store_true", help="Validation diagnostics only with mismatched/legacy provenance")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = vars(parser.parse_args(argv))
    if args["batch_size"] <= 0:
        parser.error("--batch-size must be positive")
    args["checkpoint_path"] = args.pop("checkpoint")
    evaluate(**args)


if __name__ == "__main__":
    main()
