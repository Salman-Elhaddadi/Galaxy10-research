"""Fixed-class metrics, calibration and deterministic image bootstrap."""
import numpy as np
import torch
from galaxy10.model.train import metrics_from_confusion


def classification_metrics(labels, probabilities, nll=None):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(y) or p.shape != (len(y), 10) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Expected nonempty labels and finite ten-class probabilities")
    if ((y < 0) | (y >= 10)).any():
        raise ValueError("Labels must be in 0..9")
    pred = p.argmax(1)
    matrix = np.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
    result = metrics_from_confusion(torch.from_numpy(matrix))
    losses = -np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(np.float64).tiny)) if nll is None else np.asarray(nll)
    confidence = p.max(1)
    correct = pred == y
    # [0,1/15), ... [14/15,1], including confidence exactly one.
    bins = np.minimum((confidence * 15).astype(int), 14)
    reliability = []
    ece = 0.0
    for b in range(15):
        mask = bins == b
        size = int(mask.sum())
        acc = float(correct[mask].mean()) if size else None
        conf = float(confidence[mask].mean()) if size else None
        if size:
            ece += size / len(y) * abs(acc - conf)
        reliability.append({"bin": b, "lower": b / 15, "upper": (b+1)/15, "count": size, "accuracy": acc, "confidence": conf})
    result.update(n=len(y), loss=float(losses.mean()), nll=float(losses.mean()), ece_15=float(ece),
                  brier_score=float(np.square(p - np.eye(10)[y]).sum(1).mean()),
                  mean_confidence_correct=float(confidence[correct].mean()) if correct.any() else None,
                  mean_confidence_incorrect=float(confidence[~correct].mean()) if (~correct).any() else None,
                  support=matrix.sum(1).tolist(), reliability_bins=reliability)
    return result


BOOTSTRAP_KEYS = ("macro_f1", "accuracy", "balanced_accuracy", "loss", "nll", "ece_15", "brier_score",
                  "mean_confidence_correct", "mean_confidence_incorrect", "per_class_precision", "per_class_recall", "per_class_f1")


def bootstrap(labels, probabilities, *, seed=42, resamples=1000, nll=None, reference=None, reference_nll=None):
    """Percentiles over images; reference gives paired degraded-minus-clean CIs."""
    if resamples != 1000:
        raise ValueError("Protocol requires exactly 1,000 resamples")
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels):
        raise ValueError("Cannot bootstrap empty data")
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in BOOTSTRAP_KEYS}
    for _ in range(resamples):
        idx = rng.integers(len(labels), size=len(labels))
        values = classification_metrics(labels[idx], probabilities[idx], None if nll is None else np.asarray(nll)[idx])
        base = classification_metrics(labels[idx], np.asarray(reference)[idx], None if reference_nll is None else np.asarray(reference_nll)[idx]) if reference is not None else None
        for key in draws:
            value = values[key]
            if value is not None and (base is None or base[key] is not None):
                draws[key].append(np.asarray(value) - np.asarray(base[key]) if base else value)
    return {"method": "paired_percentile_images" if reference is not None else "percentile_images",
            "confidence": 0.95, "resamples": resamples, "seed": seed,
            "intervals": {key: {"lower": np.percentile(values, 2.5, axis=0).tolist(),
                                 "upper": np.percentile(values, 97.5, axis=0).tolist(),
                                 "valid_resamples": len(values)} if values else None for key, values in draws.items()}}


def pixel_scale_metrics(labels, probabilities, scales, nll=None):
    output = {}
    for scale in np.unique(scales):
        mask = np.isclose(scales, scale, rtol=0, atol=1e-9)
        if not mask.any():
            continue
        metrics = classification_metrics(np.asarray(labels)[mask], np.asarray(probabilities)[mask], None if nll is None else np.asarray(nll)[mask])
        eligible = np.array(metrics["support"]) >= 20
        for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
            metrics[key] = [value if eligible[i] else None for i, value in enumerate(metrics[key])]
        metrics["per_class_minimum_support"] = 20
        output[str(float(scale))] = metrics
    return output
