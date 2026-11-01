"""Model definitions for the Galaxy10 DECaLS experiments."""

__all__ = ["GalAxy10", "ModelA", "CNN"]


def _load_model_attr(name, attr_name):
    try:
        module = __import__(f"{__name__}.CNN", fromlist=[attr_name])
    except ModuleNotFoundError:
        module = __import__(f"{__name__}.cnn", fromlist=[attr_name])
    return getattr(module, attr_name)


def __getattr__(name):
    if name == "GalAxy10":
        return _load_model_attr(name, "GalAxy10")
    if name == "ModelA":
        return _load_model_attr(name, "ModelA")
    if name == "CNN":
        return _load_model_attr(name, "CNN")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")