"""Aggregate saved best checkpoints; separate architectures, conditions and data."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from galaxy10.audit import ROOT, sha256, unique_results, write_json
from galaxy10.evaluate import load_checkpoint


def aggregate(checkpoints, root=ROOT, evaluations=()):
    groups = defaultdict(list)
    evaluation_by_checkpoint = defaultdict(list)
    for folder in evaluations:
        folder = Path(folder)
        if json.loads((folder / "summary.json").read_text())["status"] != "completed":
            raise ValueError(f"Incomplete evaluation: {folder}")
        cfg = json.loads((folder / "config.json").read_text())
        evaluation_by_checkpoint[cfg["checkpoint_sha256"]].append((folder, cfg))
    for path in checkpoints:
        _, checkpoint = load_checkpoint(path)
        cfg = checkpoint["config"]
        provenance = cfg.get("provenance", {}).get("file_sha256", {})
        provenance = {k.replace("\\", "/"): v for k, v in provenance.items()}
        # Source hashes differ when the condition implementation changes. Group all
        # source/protocol/input hashes to prevent accidental cross-version mixing.
        comparison = {"model": cfg["model"], "condition": cfg.get("condition", "clean"),
                      "fingerprints": provenance,
                      "settings": {k: cfg.get(k) for k in ("epochs", "batch_size", "learning_rate", "weight_decay", "optimizer", "optimizer_defaults", "loss", "scheduler", "gradient_clipping", "selection_metric", "selection_tie_break", "augmentation", "targeted_augmentation", "deterministic_algorithms", "protocol_version")}}
        key = json.dumps(comparison, sort_keys=True)
        digest = sha256(path)
        base = {"seed": cfg["seed"], "checkpoint": str(Path(path).resolve()), "checkpoint_sha256": digest,
                "best_epoch": checkpoint["epoch"]}
        groups[(key, "clean_validation", "clean", 0)].append({**base, "metrics": checkpoint["validation"]})
        for folder, evcfg in evaluation_by_checkpoint[digest]:
            evaluation_key = json.dumps({**comparison, "evaluation_data": evcfg["data_fingerprints"],
                                         "degradation_config": evcfg["degradation_config_sha256"], "evaluation_status": evcfg["status"],
                                         "evaluation_source": evcfg["evaluation_source_sha256"],
                                         "formal_evaluation": evcfg["formal_results_eligible"]}, sort_keys=True)
            for record in json.loads((folder / "metrics.json").read_text())["records"]:
                groups[(evaluation_key, evcfg["split"], record["kind"], record["severity"])].append({**base, "metrics": record["metrics"]})
    rows = []
    for (key, split, kind, severity), runs in sorted(groups.items()):
        seeds = [r["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed in a comparison group: {seeds}")
        stats = {}
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "loss", "ece_15", "brier_score", "nll", "per_class_f1"):
            if all(metric in r["metrics"] for r in runs):
                values = np.asarray([r["metrics"][metric] for r in runs], dtype=float)
                stats[metric] = {"mean": values.mean(0).tolist(), "sample_std": values.std(0, ddof=1).tolist() if len(runs) > 1 else None}
        rows.append({"comparison": json.loads(key), "split": split, "kind": kind, "severity": severity,
                     "seeds": sorted(seeds), "n_seeds": len(seeds), "complete_protocol_seed_set": sorted(seeds) == [0, 1, 2],
                     "formal_results_eligible": split == "test" and json.loads(key)["model"] == "a" and json.loads(key).get("formal_evaluation", False) and sorted(seeds) == [0, 1, 2],
                     "statistics": stats, "runs": runs})
    if not rows:
        raise ValueError("No checkpoints supplied")
    output = unique_results(root, "aggregation")
    write_json(output / "aggregation.json", {"schema_version": 1, "selection": "saved_best_clean_validation_macro_f1", "groups": rows})
    with (output / "aggregation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "model", "condition", "split", "kind", "severity", "seeds", "metric", "mean", "sample_std"])
        for row in rows:
            group = hashlib.sha256(json.dumps(row["comparison"], sort_keys=True).encode()).hexdigest()
            for metric, stats in row["statistics"].items():
                writer.writerow([group, row["comparison"]["model"], row["comparison"]["condition"], row["split"], row["kind"], row["severity"], row["seeds"], metric, json.dumps(stats["mean"]), json.dumps(stats["sample_std"])])
    print(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True, help="Explicit best.pt paths; glob expansion is shell-dependent")
    parser.add_argument("--evaluations", nargs="*", type=Path, default=[])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    aggregate(args.checkpoints, args.root, args.evaluations)


if __name__ == "__main__":
    main()
