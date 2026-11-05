import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from galaxy10.data.dataset import Galaxy10Dataset
from galaxy10.model.CNN import ModelA
from galaxy10.model.residual import ModelB
from galaxy10.data.degradations import TargetedAugmentation

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (0, 1, 2)
NUM_CLASSES = 10


@dataclass(frozen=True)
class RunSettings:
    model: str = "a"
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    batch_size: int = 32
    epochs: int = 30
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "cross_entropy_unweighted"
    scheduler: str = "none"
    gradient_clipping: str = "none"
    selection_metric: str = "clean_validation_macro_f1"
    selection_tie_break: str = "earliest_epoch"
    augmentation: str = "horizontal_flip_p0.5_vertical_flip_p0.5_uniform_rot90"
    deterministic_algorithms: bool = True
    protocol_version: str = "protocol-v1-draft-amendment-2026-11-04"
    condition: str = "clean"
    protocol_manifest: str = ""


def seed_everything(seed):
    # Set before any CUDA work. Unsupported nondeterministic operations fail loudly.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Top-level callable so Windows spawn can seed each worker."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    # PyTorch already seeds the worker's torch RNG from the loader generator.


def train_tf(image):
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(2,))
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(1,))
    return torch.rot90(image, k=int(torch.randint(0, 4, ()).item()), dims=(1, 2))


class AugmentedTransform:
    def __init__(self, seed):
        self.targeted = TargetedAugmentation(seed)

    def __call__(self, image):
        return self.targeted(train_tf(image))


def make_loader(dataset, settings, *, shuffle):
    return DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=shuffle,
        num_workers=settings.num_workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(settings.seed),
    )


def architecture_metadata(model_name):
    return {
        "architecture": "ModelA" if model_name == "a" else "ModelB",
        "experiment": "baseline_clean" if model_name == "a" else "exploratory_residual_clean",
        "architecture_protocol_eligible": model_name == "a",
        "formal_results_eligible": False,  # Section 12 remains open for the entire draft.
        "research_status": "draft_baseline" if model_name == "a" else "exploratory_excluded_from_protocol_comparisons",
    }


def create_run_paths(root, settings):
    """Reserve both directories exclusively, including against concurrent runs."""
    prefix = "model-a" if settings.model == "a" else "model-b-exploratory"
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        condition = "-aug" if settings.condition == "aug" else ""
        run_id = f"{prefix}{condition}-seed{settings.seed}-{stamp}-{uuid.uuid4().hex}"
        results = root / "results" / run_id
        checkpoints = root / "checkpoints" / run_id
        try:
            results.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        try:
            checkpoints.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            results.rmdir()  # Only the empty directory reserved by this invocation.
            continue
        return results, checkpoints


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(root, device):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    paths = list((root / "src").rglob("*.py")) + [
        root / "experiments" / "train.py", root / "pyproject.toml",
        root / "docs" / "protocol.md",
        root / "splits" / "train_idx.npy", root / "splits" / "val_idx.npy",
        root / "splits" / "group_ids.npy", root / "splits" / "excluded_idx.npy",
        root / "configs" / "data.yaml", root / "configs" / "degradations.yaml",
        root / "data" / "processed" / "galaxy10_128x128_uint8.npy",
        root / "data" / "processed" / "galaxy10_labels.npy",
        root / "data" / "processed" / "train_normalization.npz",
    ]
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "file_sha256": {str(p.relative_to(root)): sha256(p) for p in paths},
        "versions": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                     "cpu_count": os.cpu_count(), "device": device,
                     "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None},
        "torch_num_threads": torch.get_num_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "determinism_scope": "Same software, hardware and worker count; cross-platform equality is not guaranteed.",
    }


def metrics_from_confusion(confusion):
    """Macro-F1 over all ten fixed classes, with zero for undefined class F1."""
    matrix = confusion.to(torch.float64)
    tp = matrix.diag()
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {
        "macro_f1": f1.mean().item(),
        "accuracy": (tp.sum() / matrix.sum().clamp_min(1)).item(),
        "balanced_accuracy": recall[support > 0].mean().item() if (support > 0).any() else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0.0, 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.numel()
            count += labels.numel()
            pairs = labels.detach().cpu() * NUM_CLASSES + logits.detach().argmax(1).cpu()
            confusion += torch.bincount(pairs, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    if count == 0:
        raise ValueError("Cannot train or evaluate an empty dataset")
    return {"loss": loss_sum / count, **metrics_from_confusion(confusion)}


def save_best_checkpoint(path, model, optimizer, *, epoch, validation, best_score, config):
    score = validation["macro_f1"]
    if score <= best_score:
        return best_score, False
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "selection_metric": "clean_validation_macro_f1",
        "condition": config.get("condition", "clean"), "model": config.get("model"), "seed": config.get("seed"),
        "validation": validation, "config": config,
    }, temporary)
    temporary.replace(path)
    return score, True


def run_training(settings):
    if settings.seed not in SEEDS or settings.model not in ("a", "b"):
        raise ValueError("Expected model a/b and seed 0, 1 or 2")
    if settings.condition not in ("clean", "aug") or (settings.model == "b" and settings.condition != "clean"):
        raise ValueError("Targeted augmentation is defined for Model A only")
    seed_everything(settings.seed)
    # Do not load test indices here. Audit only train/validation/exclusion metadata.
    train_indices = np.load(ROOT / "splits" / "train_idx.npy", allow_pickle=False)
    val_indices = np.load(ROOT / "splits" / "val_idx.npy", allow_pickle=False)
    if not (ROOT / "splits/group_ids.npy").exists():
        raise ValueError("Missing audited groups; run scripts/make_splits.py")
    if (ROOT / "splits/group_ids.npy").exists():
        groups = np.load(ROOT / "splits/group_ids.npy", allow_pickle=False)
        excluded = np.load(ROOT / "splits/excluded_idx.npy", allow_pickle=False)
        if np.intersect1d(train_indices, val_indices).size or np.intersect1d(groups[train_indices], groups[val_indices]).size:
            raise ValueError("Train/validation leakage")
        if np.intersect1d(np.concatenate((train_indices, val_indices)), excluded).size:
            raise ValueError("Excluded image in development splits")
        from galaxy10.audit import sha256 as file_hash
        norm_meta = ROOT / "data/processed/normalization.json"
        metadata = json.loads(norm_meta.read_text()) if norm_meta.exists() else {}
        if metadata.get("train_indices_sha256") != file_hash(ROOT / "splits/train_idx.npy") or metadata.get("normalization_sha256") != file_hash(ROOT / "data/processed/train_normalization.npz"):
            raise ValueError("Missing/stale normalization provenance; rerun scripts/compute_normalisation.py")
    transform = train_tf if settings.condition == "clean" else AugmentedTransform(settings.seed)
    train_ds = Galaxy10Dataset(train_indices, transform=transform)
    val_ds = Galaxy10Dataset(np.load(ROOT / "splits" / "val_idx.npy", allow_pickle=False))
    train_loader = make_loader(train_ds, settings, shuffle=True)
    val_loader = make_loader(val_ds, settings, shuffle=False)
    model = {"a": ModelA, "b": ModelB}[settings.model]().to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    config = {**asdict(settings), **architecture_metadata(settings.model),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "num_classes": NUM_CLASSES, "macro_f1_zero_division": 0,
              "validation": "clean_no_augmentation", "test_evaluation": "not_performed",
              "optimizer_defaults": optimizer.defaults,
              "provenance": provenance(ROOT, settings.device)}
    image_hash = config["provenance"].get("file_sha256", {}).get("data/processed/galaxy10_128x128_uint8.npy")
    if image_hash is None:
        image_hash = config["provenance"].get("file_sha256", {}).get("data\\processed\\galaxy10_128x128_uint8.npy")
    if image_hash is not None and metadata.get("images_sha256") != image_hash:
        raise ValueError("Normalization image cache fingerprint is stale")
    config.update(condition=settings.condition,
                  experiment=f"model_a_{settings.condition}" if settings.model == "a" else "exploratory_model_b_clean",
                  targeted_augmentation={"probability": 0.5, "types": ["gaussian_noise", "gaussian_blur"],
                                         "type_probability": [0.5, 0.5], "noise_sigma": [0, 0.10], "blur_sigma": [0, 2.0]}
                  if settings.condition == "aug" else None)
    if settings.protocol_manifest:
        variable_keys = {"model", "condition", "seed", "device", "num_workers", "protocol_manifest"}
        defaults = asdict(RunSettings())
        if any(value != defaults[k] for k, value in asdict(settings).items() if k not in variable_keys):
            raise ValueError("Formal training settings differ from protocol defaults")
        # This reads the manifest and development-input hashes only, never test
        # indices/pixels. Full manifest integrity is checked by the evaluator.
        manifest = json.loads(Path(settings.protocol_manifest).read_text(encoding="utf-8"))
        hashes = manifest.get("fingerprints", {})
        current = config["provenance"]["file_sha256"]
        development = {k.replace("\\", "/"): v for k, v in current.items()}
        required_checks = {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "single_batch_sanity", "deliberate_test_decision"}
        if manifest.get("status") != "frozen" or not required_checks <= manifest.get("checks", {}).keys() or not all(manifest["checks"].get(k) is True and manifest.get("evidence", {}).get(k) for k in required_checks):
            raise ValueError("Training protocol manifest is not frozen")
        if any(hashes.get(k) != v for k, v in development.items() if k.startswith(("src/", "data/", "splits/", "configs/")) or k == "docs/protocol.md"):
            raise ValueError("Training inputs/source differ from frozen protocol")
        config["formal_results_eligible"] = settings.model == "a"
        config["protocol_manifest_sha256"] = sha256(Path(settings.protocol_manifest))
        config["research_status"] = "frozen_protocol_model_a" if settings.model == "a" else config["research_status"]
    results, checkpoints = create_run_paths(ROOT, settings)
    config.update(run_id=results.name, results_path=str(results), checkpoint_path=str(checkpoints / "best.pt"))
    write_json(results / "config.json", config)
    print(f"Run {results.name}: {config['research_status']}", flush=True)
    history, best_score, best_epoch = [], -1.0, None
    write_json(results / "summary.json", {"status": "running", "run_id": results.name, "condition": settings.condition, "model": settings.model, "seed": settings.seed})
    try:
        for epoch in range(1, settings.epochs + 1):
            train = run_epoch(model, train_loader, settings.device, optimizer)
            validation = run_epoch(model, val_loader, settings.device)
            best_score, selected = save_best_checkpoint(
                checkpoints / "best.pt", model, optimizer, epoch=epoch, validation=validation,
                best_score=best_score, config=config,
            )
            if selected:
                best_epoch = epoch
            history.append({"epoch": epoch, "model": settings.model, "condition": settings.condition, "seed": settings.seed, "train": train, "clean_validation": validation, "selected": selected})
            write_json(results / "metrics.json", history)
            print(f"Epoch {epoch}/{settings.epochs} | train loss {train['loss']:.4f} | "
                  f"clean val macro-F1 {validation['macro_f1']:.4f} | val accuracy {validation['accuracy']:.4f}", flush=True)
    except BaseException as error:
        write_json(results / "summary.json", {"status": "failed", "error": str(error), "run_id": results.name, "condition": settings.condition, "model": settings.model, "seed": settings.seed})
        raise
    write_json(results / "summary.json", {
        "status": "completed", "run_id": results.name, "seed": settings.seed,
        "condition": settings.condition,
        **architecture_metadata(settings.model), "best_epoch": best_epoch,
        "best_clean_validation_macro_f1": best_score, "checkpoint_path": str(checkpoints / "best.pt"),
        "formal_results_eligible": config["formal_results_eligible"],
        "research_status": config["research_status"],
        "experiment": config["experiment"],
        "test_evaluation": "not_performed",
    })
    return results, checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean-data training under the draft protocol; Model B is exploratory.")
    parser.add_argument("--model", choices=("a", "b"), default="a")
    parser.add_argument("--condition", choices=("clean", "aug"), default="clean")
    parser.add_argument("--protocol-manifest", default="", help="After deliberate protocol freezing only; training never reads test inputs")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0, help="Repeat with 0, 1 and 2 for each configuration.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.model == "b" and args.condition == "aug":
        parser.error("--condition aug is Model A only; Model B remains exploratory")
    run_training(RunSettings(model=args.model, seed=args.seed, device=device, num_workers=args.num_workers, condition=args.condition, protocol_manifest=args.protocol_manifest))


if __name__ == "__main__":
    main()
