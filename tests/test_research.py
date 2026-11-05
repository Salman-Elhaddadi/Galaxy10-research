"""Research invariants and end-to-end workflows on synthetic CPU data only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch

from galaxy10.audit import authorize_split, data_fingerprints, sha256
from galaxy10.data.degradations import degrade, TargetedAugmentation
from galaxy10.data.splits import build_groups, generate_splits, validate_splits, load_splits
from galaxy10.evaluate import evaluate, load_checkpoint
from galaxy10.gradcam import gradcam, select_examples, run_gradcam
from galaxy10.aggregate import aggregate
from galaxy10.metrics import bootstrap, classification_metrics
from galaxy10.model.CNN import ModelA
from galaxy10.model.train import seed_everything, train_tf


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_transitive_duplicate_spatial_groups_and_conflicting_exclusions():
    labels = np.array([0, 1, 0, 2, 2, 3])
    digests = np.array(["a", "a", "b", "c", "c", "d"])
    # 0/1 duplicate; 1/2 sky match; 3/4 same-label duplicate.
    ra = np.array([100., 200., 200.0001, 300., 301., 350.])
    groups, excluded, conflicts, _ = build_groups(labels, digests, ra, np.zeros(6))
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] == groups[4]
    assert excluded.tolist() == [0, 1]
    assert conflicts == [[0, 1]]
    with pytest.raises(ValueError, match="Group crosses"):
        validate_splits({"train": np.array([2, 3]), "val": np.array([4]), "test": np.array([5])}, excluded, groups, 6)
    with pytest.raises(ValueError, match="overlap"):
        validate_splits({"train": np.array([0, 2]), "val": np.array([3, 4]), "test": np.array([5])}, excluded, groups, 6)


def test_nested_stratified_split_repeats_and_never_crosses_groups():
    labels = np.repeat(np.arange(10), 100)
    groups = np.arange(1000) // 2
    excluded = np.array([0, 1], dtype=np.int64)
    a = generate_splits(labels, groups, excluded)
    b = generate_splits(labels, groups, excluded)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert validate_splits(a, excluded, groups, len(labels))["groups_crossing_splits"] == 0
    assert all(np.all(np.bincount(labels[v], minlength=10) > 0) for v in a.values())


def test_preparation_report_is_byte_reproducible_and_archives_original(tmp_path):
    import h5py
    import importlib.util
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prepare_splits", repo / "scripts/make_splits.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path / "configs").mkdir()
    shutil.copy(repo / "configs/data.yaml", tmp_path / "configs/data.yaml")
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/processed").mkdir()
    labels = np.repeat(np.arange(10), 30)
    np.save(tmp_path / "data/processed/galaxy10_labels.npy", labels)
    with h5py.File(tmp_path / "data/raw/Galaxy10_DECals.h5", "w") as handle:
        handle["images"] = np.random.default_rng(1).integers(0,256,(300,8,8,3),dtype=np.uint8)
        handle["ans"] = labels
        handle["ra"] = np.arange(300, dtype=float)
        handle["dec"] = np.zeros(300)
        handle["pxscale"] = np.full(300, .262)
    module.prepare(tmp_path)
    before = (tmp_path / "splits/split_report.json").read_bytes()
    module.prepare(tmp_path)
    assert (tmp_path / "splits/split_report.json").read_bytes() == before
    assert list((tmp_path / "splits/history").glob("*/split_report.json"))
    load_splits(tmp_path)


def test_committed_split_isolation_and_fingerprints():
    root = Path(__file__).resolve().parents[1]
    # Dataset-free: indices/groups are committed. Full label cache is not needed.
    folder = root / "splits"
    report = json.loads((folder / "split_report.json").read_text())
    splits = {k: np.load(folder / f"{k}_idx.npy", allow_pickle=False) for k in ("train", "val", "test")}
    validate_splits(splits, np.load(folder / "excluded_idx.npy"), np.load(folder / "group_ids.npy"), report["n_images"])
    for name, digest in report["output_sha256"].items():
        assert sha256(folder / name) == digest


@pytest.mark.parametrize("kind,level", [("gaussian_noise", .1), ("gaussian_blur", 2), ("resolution", 4), ("contrast", .25)])
def test_degradations_repeat_across_order_and_are_clipped(kind, level):
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32)
    before = torch.random.get_rng_state().clone()
    a = degrade(image, kind, level, seed=42, index=7)
    degrade(image, kind, level, seed=42, index=9)
    b = degrade(image, kind, level, seed=42, index=7)
    assert torch.equal(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.shape == image.shape and 0 <= a.min() <= a.max() <= 1
    if kind == "gaussian_noise":
        assert not torch.equal(a, degrade(image, kind, level, seed=42, index=8))
        assert not torch.equal(a, degrade(image, kind, .05, seed=42, index=7))


def test_degradation_definitions_and_identity():
    image = torch.rand(3, 32, 32)
    for kind, value in [("gaussian_noise", 0), ("gaussian_blur", 0), ("resolution", 1), ("contrast", 1)]:
        assert torch.allclose(degrade(image, kind, value), image, atol=1e-7)
    assert torch.allclose(degrade(image, "contrast", 0), torch.full_like(image, image.mean()))
    blurred = degrade(torch.ones_like(image) * .4, "gaussian_blur", 2)
    assert torch.allclose(blurred, torch.full_like(image, .4), atol=1e-6)
    with pytest.raises(ValueError):
        degrade(image, "resolution", 3)


def test_targeted_policy_exactly_one_type_and_geometry_rng_preserved(monkeypatch):
    from galaxy10.data import degradations as module
    calls = []
    monkeypatch.setattr(module, "degrade", lambda image, kind, severity, **kwargs: calls.append((kind, severity)) or image)
    image = torch.arange(3*32*32).reshape(3,32,32)
    seed_everything(0)
    expected = [train_tf(image) for _ in range(2000)]
    seed_everything(0)
    augment = TargetedAugmentation(0)
    actual = [augment(train_tf(image)) for _ in range(2000)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert 900 < len(calls) < 1100
    noise = [(kind, severity) for kind, severity in calls if kind == "gaussian_noise"]
    assert .4 < len(noise) / len(calls) < .6
    assert all(0 <= value <= (.1 if kind == "gaussian_noise" else 2) for kind, value in calls)


def test_metrics_against_hand_calculation_and_bootstrap():
    p = np.zeros((2, 10)); p[0, :2] = [.8, .2]; p[1, :2] = [.6, .4]
    labels = np.array([0, 1])
    result = classification_metrics(labels, p)
    assert result["accuracy"] == .5
    assert result["macro_f1"] == pytest.approx((2/3)/10)
    assert result["balanced_accuracy"] == .5
    assert result["ece_15"] == pytest.approx(.4)
    assert result["brier_score"] == pytest.approx((.08 + .72)/2)
    assert result["nll"] == pytest.approx((-np.log(.8)-np.log(.4))/2)
    assert result["mean_confidence_correct"] == .8
    assert result["mean_confidence_incorrect"] == .6
    first = bootstrap(labels, p)
    assert first == bootstrap(labels, p)
    assert first["resamples"] == 1000
    perfect = np.eye(10)
    calibrated = classification_metrics(np.arange(10), perfect)
    assert calibrated["ece_15"] == calibrated["brier_score"] == calibrated["nll"] == 0
    assert calibrated["reliability_bins"][-1]["count"] == 10
    ci = bootstrap(np.arange(10), perfect)
    assert ci["intervals"]["accuracy"]["lower"] == ci["intervals"]["accuracy"]["upper"] == 1
    paired = bootstrap(labels, p, reference=p)
    assert paired["intervals"]["per_class_f1"]["lower"] == [0] * 10


def test_test_access_protection_precedes_any_data_read(tmp_path):
    with pytest.raises(ValueError, match="--frozen-test"):
        evaluate(tmp_path, tmp_path / "absent.pt", "test")
    with pytest.raises(ValueError, match="--freeze-manifest"):
        evaluate(tmp_path, tmp_path / "absent.pt", "test", frozen_test=True)
    manifest = tmp_path / "draft.json"
    manifest.write_text(json.dumps({"status": "draft"}))
    with pytest.raises(ValueError, match="unfrozen"):
        authorize_split(tmp_path, "test", True, manifest)


def test_paired_bootstrap_preserves_log_space_nll_under_probability_underflow():
    p = np.eye(10)[[0]]
    ci = bootstrap(np.array([1]), p, nll=np.array([1000.0]), reference=p, reference_nll=np.array([2000.0]))
    assert ci["intervals"]["nll"]["lower"] == ci["intervals"]["nll"]["upper"] == -1000


def test_freeze_fingerprint_tampering_is_rejected(tmp_path, monkeypatch):
    from galaxy10 import audit
    checks = {**{f"P{i}": True for i in range(1,9)}, "single_batch_sanity": True, "deliberate_test_decision": True}
    manifest = {"status": "frozen", "checks": checks, "evidence": {k: "synthetic fixture" for k in checks}, "fingerprints": {"fixture": "abc"}}
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, "freeze_fingerprints", lambda root, **kwargs: {"fixture": "abc"})
    assert authorize_split(tmp_path, "test", True, path) == manifest
    monkeypatch.setattr(audit, "freeze_fingerprints", lambda root, **kwargs: {"fixture": "changed"})
    with pytest.raises(ValueError, match="fingerprints"):
        authorize_split(tmp_path, "test", True, path)


@pytest.fixture
def synthetic_repository(tmp_path):
    root = tmp_path / "synthetic"
    processed, splits = root / "data/processed", root / "splits"
    processed.mkdir(parents=True); splits.mkdir()
    (root / "configs").mkdir()
    shutil.copy(Path(__file__).resolve().parents[1] / "configs/degradations.yaml", root / "configs/degradations.yaml")
    # Two validation images, no real Galaxy10 pixels or labels.
    images = np.random.default_rng(3).integers(0, 256, (6,128,128,3), dtype=np.uint8)
    np.save(processed / "galaxy10_128x128_uint8.npy", images)
    np.save(processed / "galaxy10_labels.npy", np.arange(6, dtype=np.int64))
    np.savez(processed / "train_normalization.npz", mean=np.array([.5]*3), std=np.array([.2]*3))
    for name, values in {"train": [0,1], "val": [2,3], "test": [4,5], "excluded": []}.items():
        np.save(splits / f"{name}_idx.npy", np.array(values, dtype=np.int64))
    np.save(splits / "group_ids.npy", np.arange(6))
    np.save(splits / "pixel_scale.npy", np.array([.262, .524]*3))
    seed_everything(0)
    model = ModelA()
    config = {"model": "a", "seed": 0, "condition": "clean", "parameter_count": 391466,
              "provenance": {"file_sha256": data_fingerprints(root)}}
    checkpoint = root / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 2, "selection_metric": "clean_validation_macro_f1",
                "validation": {"macro_f1": .2, "accuracy": .5}, "config": config}, checkpoint)
    return root, checkpoint


def test_checkpoint_strict_compatibility_and_legacy_rejection(synthetic_repository, tmp_path):
    root, path = synthetic_repository
    model, checkpoint = load_checkpoint(path)
    assert isinstance(model, ModelA) and checkpoint["epoch"] == 2
    bad = tmp_path / "raw.pt"
    torch.save(model.state_dict(), bad)
    with pytest.raises(ValueError, match="saved best"):
        load_checkpoint(bad)
    np.save(root / "data/processed/galaxy10_labels.npy", np.arange(6)[::-1])
    with pytest.raises(ValueError, match="fingerprints"):
        evaluate(root, path, "val")


def test_augmented_model_a_training_artifacts_on_synthetic_images(synthetic_repository, monkeypatch):
    from galaxy10.model import train
    root, _ = synthetic_repository
    processed = root / "data/processed"
    (processed / "normalization.json").write_text(json.dumps({
        "train_indices_sha256": sha256(root / "splits/train_idx.npy"),
        "images_sha256": sha256(processed / "galaxy10_128x128_uint8.npy"),
        "normalization_sha256": sha256(processed / "train_normalization.npz")}))
    monkeypatch.setattr(train, "ROOT", root)
    original_dataset = train.Galaxy10Dataset
    monkeypatch.setattr(train, "Galaxy10Dataset", lambda indices, transform=None: original_dataset(indices, transform=transform, root=root))
    monkeypatch.setattr(train, "provenance", lambda root, device: {"file_sha256": data_fingerprints(root)})
    result, path = train.run_training(train.RunSettings(condition="aug", epochs=1))
    config = json.loads((result / "config.json").read_text())
    assert config["condition"] == "aug" and config["experiment"] == "model_a_aug"
    assert config["targeted_augmentation"]["probability"] == .5
    assert load_checkpoint(path / "best.pt")[1]["epoch"] == 1
    assert json.loads((result / "summary.json").read_text())["condition"] == "aug"


def test_synthetic_frozen_access_manifest_self_hash_and_once_ledger(synthetic_repository):
    from galaxy10.audit import freeze_fingerprints
    root, checkpoint = synthetic_repository
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "docs/protocol.md").write_text("Synthetic fixture only; not a real protocol freeze")
    (root / "splits/split_report.json").write_text("{}")
    checks = {**{f"P{i}": True for i in range(1,9)}, "single_batch_sanity": True, "deliberate_test_decision": True}
    manifest = root / "configs/frozen_protocol.json"
    manifest.write_text(json.dumps({"status": "frozen", "checks": checks, "evidence": {k: "synthetic fixture only" for k in checks},
                                    "fingerprints": freeze_fingerprints(root, exclude=[manifest])}))
    output = evaluate(root, checkpoint, "test", frozen_test=True, freeze_manifest=manifest)
    assert json.loads((output / "summary.json").read_text())["test_access"]
    assert not json.loads((output / "config.json").read_text())["formal_results_eligible"]
    with pytest.raises(FileExistsError):
        evaluate(root, checkpoint, "test", frozen_test=True, freeze_manifest=manifest)
    assert len(list((root / "results/test_access").glob("*.json"))) == 1


def test_normalization_uses_no_test_indices_and_rejects_exclusions(synthetic_repository):
    import importlib.util
    root, _ = synthetic_repository
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("normalization", repo / "scripts/compute_normalisation.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (root / "splits/test_idx.npy").unlink()  # Synthetic fixture only.
    module.compute(root)
    metadata = json.loads((root / "data/processed/normalization.json").read_text())
    assert metadata["test_access"] is False and metadata["n_train"] == 2
    np.save(root / "splits/excluded_idx.npy", np.array([0]))
    with pytest.raises(ValueError, match="isolation"):
        module.compute(root)


def test_aggregation_sample_std_and_condition_isolation(synthetic_repository):
    root, path = synthetic_repository
    original = torch.load(path, weights_only=True)
    paths = []
    for condition in ("clean", "aug"):
        for seed in (0,1,2):
            checkpoint = {**original, "config": {**original["config"], "condition": condition, "seed": seed},
                          "validation": {"macro_f1": .1 + seed*.1, "accuracy": .5}, "epoch": 2+seed}
            target = root / f"{condition}-{seed}-best.pt"
            torch.save(checkpoint, target)
            paths.append(target)
    output = aggregate(paths, root)
    groups = json.loads((output / "aggregation.json").read_text())["groups"]
    assert len(groups) == 2
    assert {g["comparison"]["condition"] for g in groups} == {"clean", "aug"}
    assert all(g["statistics"]["macro_f1"]["mean"] == pytest.approx(.2) for g in groups)
    assert all(g["statistics"]["macro_f1"]["sample_std"] == pytest.approx(.1) for g in groups)


def test_full_synthetic_validation_degradation_gradcam_aggregation(synthetic_repository):
    root, checkpoint = synthetic_repository
    original = sha256(checkpoint)
    output = evaluate(root, checkpoint, "val", degradations=True)
    assert sha256(checkpoint) == original
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["status"] == "validation_diagnostic" and cfg["preliminary_mismatches"] == []
    records = json.loads((output / "metrics.json").read_text())["records"]
    assert len(records) == 20
    for record in records:
        folder = output / record["name"]
        assert (folder / "predictions.csv").is_file()
        assert record["confidence_intervals"]["resamples"] == 1000
        assert len(record["metrics"]["per_class_f1"]) == 10
        with np.load(folder / "predictions.npz") as predictions:
            assert predictions["indices"].tolist() == [2,3]
            assert predictions["probabilities"].shape == (2,10)
        assert len(record["metrics"]["by_pixel_scale"]) == 2
    assert len(list(output.glob("*.png"))) == 13
    cams = run_gradcam(output, root=root, per_class=1)
    selection = json.loads((cams / "selection.json").read_text())
    assert len(selection["selected"]) == 2 and selection["seed"] == 42
    assert len(list(cams.glob("*.npy"))) == 20
    agg = aggregate([checkpoint], root, [output])
    groups = json.loads((agg / "aggregation.json").read_text())["groups"]
    assert len(groups) == 21 and all(g["n_seeds"] == 1 for g in groups)
    assert all(g["statistics"]["macro_f1"]["sample_std"] is None for g in groups)
    with pytest.raises(ValueError, match="Duplicate seed"):
        aggregate([checkpoint, checkpoint], root)


def test_gradcam_target_shape_and_preselection():
    seed_everything(0)
    model = ModelA().eval()
    cam, target = gradcam(model, torch.rand(3,128,128))
    assert cam.shape == (128,128) and np.isfinite(cam).all() and 0 <= target < 10
    indices, labels = np.arange(20), np.repeat(np.arange(10), 2)
    pred = labels.copy(); pred[::2] = (pred[::2] + 1) % 10
    assert select_examples(indices, labels, pred) == select_examples(indices, labels, pred)
    assert len(select_examples(indices, labels, pred)) == 20


@pytest.mark.parametrize("entry", ["experiments/evaluate.py", "experiments/aggregate.py", "experiments/gradcam.py", "experiments/run_matrix.py", "scripts/make_splits.py", "scripts/compute_normalisation.py", "scripts/audit_preparation.py", "scripts/freeze_protocol.py", "scripts/prepare_kaggle.py"])
def test_entrypoints_from_parent(entry):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / entry), "--help"], cwd=root.parent, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
