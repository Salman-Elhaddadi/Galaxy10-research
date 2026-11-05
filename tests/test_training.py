"""Small CPU checks; no Galaxy10 data or research training runs required."""

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from galaxy10.model import train
from galaxy10.model.CNN import ModelA
from galaxy10.model.residual import ModelB


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RandomDataset(Dataset):
    """Exercise Python, NumPy, torch and geometric RNGs inside real workers."""

    def __len__(self):
        return 12

    def __getitem__(self, index):
        image = train.train_tf(torch.arange(27).reshape(3, 3, 3))
        return index, random.random(), np.random.random(), torch.rand(()), image


def test_model_b_shape_and_exploratory_exclusion():
    model = ModelB().eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 128, 128)).shape == (2, 10)
    count_a = sum(p.numel() for p in ModelA().parameters())
    count_b = sum(p.numel() for p in model.parameters())
    assert count_b > 2 * count_a
    assert not train.architecture_metadata("b")["architecture_protocol_eligible"]
    assert not train.architecture_metadata("b")["formal_results_eligible"]
    assert train.architecture_metadata("a")["architecture_protocol_eligible"]


def test_protocol_settings_identical_across_architectures():
    a = train.RunSettings(model="a")
    b = train.RunSettings(model="b")
    assert a == replace(b, model="a")
    assert (a.optimizer, a.learning_rate, a.weight_decay) == ("Adam", 1e-3, 0)
    assert (a.batch_size, a.epochs, a.gradient_clipping) == (32, 30, "none")


def test_macro_f1_uses_all_classes_and_zero_division():
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    matrix[0, 0], matrix[0, 1], matrix[1, 1] = 8, 2, 2
    metrics = train.metrics_from_confusion(matrix)
    assert metrics["macro_f1"] == pytest.approx((16 / 18 + 4 / 6) / 10)
    assert metrics["accuracy"] == pytest.approx(10 / 12)
    assert metrics["per_class_f1"][2:] == [0.0] * 8


def test_checkpoint_uses_macro_f1_even_when_accuracy_decreases(tmp_path):
    model = nn.Linear(1, 10)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "best.pt"
    best = -1.0
    # Accuracy falls as F1 rises; then accuracy rises while F1 falls; finally an F1 tie.
    for epoch, (f1, accuracy) in enumerate([(0.1, 0.9), (0.3, 0.7), (0.2, 0.95), (0.3, 0.99)], 1):
        with torch.no_grad():
            model.bias.fill_(epoch)
        best, selected = train.save_best_checkpoint(
            path, model, optimizer, epoch=epoch,
            validation={"macro_f1": f1, "accuracy": accuracy}, best_score=best, config={"seed": 0},
        )
        assert selected == (epoch in (1, 2))
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["validation"]["macro_f1"] == best == 0.3
    assert torch.all(checkpoint["model_state_dict"]["bias"] == 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seed_repeats_python_numpy_torch_initialization_and_augmentation(seed):
    def sample():
        train.seed_everything(seed)
        return (random.random(), np.random.rand(), ModelA().classifier[-1].weight.detach().clone(),
                torch.stack([train.train_tf(torch.arange(27).reshape(3, 3, 3)) for _ in range(8)]))
    a, b = sample(), sample()
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[3], b[3])
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_repeats_shuffling_and_worker_augmentation(workers):
    def collect(seed):
        train.seed_everything(seed)
        loader = train.make_loader(RandomDataset(), train.RunSettings(seed=seed, num_workers=workers), shuffle=True)
        # Also verify the next epoch's advancing generator/worker streams repeat.
        return [tuple(t.clone() for t in next(iter(loader))) for _ in range(2)]
    first, second, different = collect(0), collect(0), collect(1)
    for a, b in zip(first, second):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(first[0][0], different[0][0])
    assert not torch.equal(first[0][0], first[1][0])


def test_geometric_augmentation_preserves_pixels():
    train.seed_everything(0)
    image = torch.arange(3 * 16 * 16).reshape(3, 16, 16)
    for _ in range(12):
        augmented = train.train_tf(image)
        assert augmented.shape == image.shape
        for channel in range(3):
            assert torch.equal(augmented[channel].flatten().sort().values, image[channel].flatten().sort().values)


def test_run_paths_never_overwrite_same_model_seed(tmp_path):
    paths = [train.create_run_paths(tmp_path, train.RunSettings()) for _ in range(8)]
    assert len({result.name for result, _ in paths}) == 8
    for result, checkpoint in paths:
        assert result.name == checkpoint.name
        assert result.parent == tmp_path / "results"
        assert checkpoint.parent == tmp_path / "checkpoints"
        assert result.is_dir() and checkpoint.is_dir()
    original = paths[0][1] / "best.pt"
    original.write_bytes(b"existing checkpoint")
    train.create_run_paths(tmp_path, train.RunSettings())
    assert original.read_bytes() == b"existing checkpoint"
    result, _ = train.create_run_paths(tmp_path, train.RunSettings(model="b", seed=2))
    assert result.name.startswith("model-b-exploratory-seed2-")


def test_run_path_collision_preserves_existing_checkpoint(tmp_path, monkeypatch):
    class FixedClock:
        @staticmethod
        def now(tz):
            return train.datetime(2026, 11, 4, tzinfo=tz)

    # Keep the real datetime available while substituting the clock.
    real_datetime = train.datetime
    FixedClock.now = staticmethod(lambda tz: real_datetime(2026, 11, 4, tzinfo=tz))
    monkeypatch.setattr(train, "datetime", FixedClock)
    from types import SimpleNamespace
    ids = iter(["collision", "fresh"])
    monkeypatch.setattr(train.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    existing = tmp_path / "checkpoints" / "model-a-seed0-20261104T000000000000Z-collision"
    existing.mkdir(parents=True)
    (existing / "best.pt").write_bytes(b"keep me")
    result, checkpoint = train.create_run_paths(tmp_path, train.RunSettings())
    assert result.name.endswith("fresh") and checkpoint.name == result.name
    assert (existing / "best.pt").read_bytes() == b"keep me"
    assert not (tmp_path / "results" / existing.name).exists()


def test_validation_is_repeatable_and_sample_weighted():
    logits = torch.zeros(3, 10)
    logits[0, 0] = 8
    logits[1, 1] = 2
    logits[2, 1] = 5
    labels = torch.tensor([0, 1, 2])
    loader = torch.utils.data.DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = nn.Identity()
    expected = nn.functional.cross_entropy(logits, labels).item()
    first = train.run_epoch(model, loader, "cpu")
    assert first == train.run_epoch(model, loader, "cpu")
    assert first["loss"] == pytest.approx(expected)
    assert not model.training


def test_tiny_training_writes_artifacts_without_reading_test_set(tmp_path, monkeypatch):
    (tmp_path / "splits").mkdir()
    np.save(tmp_path / "splits" / "train_idx.npy", np.arange(4))
    np.save(tmp_path / "splits" / "val_idx.npy", np.arange(4, 8))
    np.save(tmp_path / "splits" / "group_ids.npy", np.arange(8))
    np.save(tmp_path / "splits" / "excluded_idx.npy", np.array([], dtype=np.int64))
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    np.savez(processed / "train_normalization.npz", mean=np.zeros(3), std=np.ones(3))
    (processed / "normalization.json").write_text(json.dumps({
        "train_indices_sha256": train.sha256(tmp_path / "splits/train_idx.npy"),
        "normalization_sha256": train.sha256(processed / "train_normalization.npz")}))
    # No test_idx.npy: any accidental test access fails the test.
    transforms = []

    def dataset(indices, transform=None):
        transforms.append(transform)
        return TensorDataset(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12, torch.arange(4))

    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "Galaxy10Dataset", dataset)
    monkeypatch.setattr(train, "ModelA", lambda: nn.Linear(3, 10))
    monkeypatch.setattr(train, "provenance", lambda root, device: {"fixture": "synthetic"})
    settings = train.RunSettings(epochs=2)
    result, checkpoint = train.run_training(settings)
    config = json.loads((result / "config.json").read_text())
    metrics = json.loads((result / "metrics.json").read_text())
    summary = json.loads((result / "summary.json").read_text())
    saved = torch.load(checkpoint / "best.pt", weights_only=True)
    assert transforms == [train.train_tf, None]
    assert config["seed"] == 0
    assert config["optimizer_defaults"]["lr"] == 1e-3
    assert config["optimizer_defaults"]["weight_decay"] == 0
    assert len(metrics) == 2
    assert summary["status"] == "completed"
    assert summary["research_status"] == "draft_baseline"
    assert summary["test_evaluation"] == "not_performed"
    assert saved["epoch"] == summary["best_epoch"]
    assert saved["config"]["run_id"] == result.name
    # A second identical synthetic run has different paths and identical results.
    result2, checkpoint2 = train.run_training(settings)
    assert result2 != result and checkpoint2 != checkpoint
    assert json.loads((result2 / "metrics.json").read_text()) == metrics


@pytest.mark.parametrize("from_parent", [False, True])
def test_documented_script_from_root_and_parent(from_parent):
    root = Path(__file__).resolve().parents[1]
    cwd = root.parent if from_parent else root
    script = Path(root.name) / "experiments" / "train.py" if from_parent else Path("experiments/train.py")
    completed = subprocess.run([sys.executable, str(script), "--help"], cwd=cwd, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "--seed {0,1,2}" in completed.stdout
    assert "exploratory" in completed.stdout


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cli_passes_requested_seed_without_training(seed, monkeypatch):
    received = []
    monkeypatch.setattr(train, "run_training", received.append)
    train.main(["--model", "b", "--seed", str(seed), "--device", "cpu"])
    assert received == [train.RunSettings(model="b", seed=seed)]
