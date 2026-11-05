"""Checkout entry point for seed aggregation."""
from pathlib import Path
import sys
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from galaxy10.aggregate import main
    main()
