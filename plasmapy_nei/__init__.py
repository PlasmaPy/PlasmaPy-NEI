"""A Python package for non-equilibrium ionization modelingg of plasma."""

import warnings

try:
    from .version import __version__

    del version
except Exception as exc:
    warnings.warn("Unable to import __version__")
finally:
    del warnings

from . import eigen
from . import nei

__all__ = ["eigen", "nei"]
