"""A Python package for non-equilibrium ionization modelingg of plasma."""

import warnings

try:
    from .version import __version__
except Exception as exc:
    warnings.warn("Unable to import __version__")
else:
    del version
finally:
    del warnings

from . import eigen
from . import nei

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["eigen"]
