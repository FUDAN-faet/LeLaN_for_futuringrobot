"""Compatibility alias for the historical ``lelan`` package name."""

from importlib import import_module
import sys


_VINT_TRAIN = import_module("vint_train")

__all__ = getattr(_VINT_TRAIN, "__all__", [])
__doc__ = _VINT_TRAIN.__doc__
__path__ = getattr(_VINT_TRAIN, "__path__", [])


def __getattr__(name):
    return getattr(_VINT_TRAIN, name)


for _submodule in ("data", "models", "process_data", "training", "visualizing"):
    _module = import_module(f"vint_train.{_submodule}")
    globals()[_submodule] = _module
    sys.modules[f"{__name__}.{_submodule}"] = _module
