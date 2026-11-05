"""Checkout entry point for preselected Grad-CAM."""
from pathlib import Path
import sys
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from galaxy10.gradcam import main
    main()
