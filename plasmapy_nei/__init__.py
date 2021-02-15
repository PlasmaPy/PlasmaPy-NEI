"""A Python package for non-equilibrium ionization modelingg of plasma."""

__all__ = ["eigen", "nei"]

import warnings

try:
    from .version import __version__
except Exception as exc:
    warnings.warn("Unable to import __version__")
finally:
    del warnings

from . import eigen, nei
