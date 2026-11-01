"""Model definitions for the Galaxy10 DECaLS experiments."""

__all__ = ["GalAxy10"]


def __getattr__(name):
    if name == "GalAxy10":
        from .CNN import GalAxy10

        return GalAxy10
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")