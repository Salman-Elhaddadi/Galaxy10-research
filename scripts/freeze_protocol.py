"""Deliberately freeze verified prerequisites; never invent evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from galaxy10.audit import freeze_fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checklist", type=Path, required=True, help="JSON checks and evidence mappings for P1..P8, single_batch_sanity, deliberate_test_decision")
    parser.add_argument("--output", type=Path, required=True, help="New manifest path (exclusive creation)")
    parser.add_argument("--confirm-frozen", action="store_true", required=True)
    args = parser.parse_args(argv)
    value = json.loads(args.checklist.read_text(encoding="utf-8"))
    required = [*[f"P{i}" for i in range(1, 9)], "single_batch_sanity", "deliberate_test_decision"]
    if not all(value.get("checks", {}).get(k) is True and value.get("evidence", {}).get(k) for k in required):
        parser.error("Every prerequisite must have a true check and a concrete evidence reference")
    # Verify saved split partition/group integrity before recording the freeze.
    from galaxy10.data.splits import load_splits
    load_splits(args.root)
    value.update(schema_version=1, status="frozen", fingerprints=freeze_fingerprints(args.root, exclude=[args.output]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
