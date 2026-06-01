"""
Sonata: a non-negative matrix factorization toolkit for signature analysis
==========================================================================
"""

from . import models
from . import plot as pl
from . import tools as tl

__version__ = "0.1.0"

pl.set_sonata_style()

__all__ = [
    "__version__",
    "models",
    "pl",
    "tl",
]
