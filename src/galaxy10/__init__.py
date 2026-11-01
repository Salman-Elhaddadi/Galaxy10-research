"""Galaxy10 DECaLS robustness research package."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Galaxy10Dataset",
    "load_splits",
    "GalAxy10",
]


def __getattr__(name):
    if name == "Galaxy10Dataset":
        from .data.dataset import Galaxy10Dataset

        return Galaxy10Dataset
    if name == "load_splits":
        from .data.splits import load_splits

        return load_splits
    if name == "GalAxy10":
        from .model.CNN import GalAxy10

        return GalAxy10
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")