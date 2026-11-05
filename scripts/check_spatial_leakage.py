"""Validate all committed groups, exclusions and partitions without image reads."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from galaxy10.data.splits import load_splits

if __name__ == "__main__":
    splits = load_splits(ROOT)
    print({name: len(indices) for name, indices in splits.items()})
    print("Complete partition; excluded images absent; no duplicate/sky group crosses splits.")
