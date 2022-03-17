"""A Python package for non-equilibrium ionization modelingg of plasma."""

__all__ = ["eigen", "nei"]

import warnings

try:
    from plasmapy_nei.version import __version__
except Exception:
    warnings.warn("Unable to import __version__")
finally:
    del warnings

from plasmapy_nei import eigen, nei
