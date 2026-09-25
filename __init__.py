"""spdee.solvers.bbjd -- JDQZ (Jacobi-Davidson QZ) eigensolver, C core.

Build the extension in place from this directory:
    python setup.py build_ext --inplace
See README.md.  ``jdqz`` is re-exported here; without the built extension
the package imports and ``jdqz`` raises when called.
"""

from .bbjd import jdqz, _bbjd_c  # noqa: F401

HAVE_C_EXT = _bbjd_c is not None
__all__ = ["jdqz", "HAVE_C_EXT"]
