"""Run the Galaxy10 training experiment from a repository checkout."""

from pathlib import Path
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from galaxy10.model.train import main

    main()
