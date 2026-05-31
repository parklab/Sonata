"""
A collection of NMF algorithms
"""

from .corrnmf import CorrNMF
from .klnmf import KLNMF
from .mvnmf import MvNMF

__all__ = [
    "CorrNMF",
    "KLNMF",
    "MvNMF",
]
