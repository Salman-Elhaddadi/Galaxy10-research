"""Dataset utilities for the Galaxy10 DECaLS experiments."""

__all__ = ["Galaxy10Dataset", "load_normalization_stats", "normalize_image", "load_splits"]


def __getattr__(name):
    if name == "Galaxy10Dataset":
        from .dataset import Galaxy10Dataset

        return Galaxy10Dataset
    if name == "load_normalization_stats":
        from .preprocessing import load_normalization_stats

        return load_normalization_stats
    if name == "normalize_image":
        from .preprocessing import normalize_image

        return normalize_image
    if name == "load_splits":
        from .splits import load_splits

        return load_splits
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")