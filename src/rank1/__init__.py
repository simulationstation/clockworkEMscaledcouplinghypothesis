"""
Rank-1 Factorization Test Pipeline for Particle Physics Datasets.

This package provides tools for testing rank-1 factorization hypotheses
on matrix-structured physics data, including:
- ATLAS Higgs production x decay signal strengths
- TOTEM elastic pp scattering dσ/dt across energies
- H1/ZEUS diffractive DIS structure functions

Key components:
- Data acquisition from HEPData, CERN Open Data, and arXiv
- Low-rank matrix fitting (ALS and nonlinear least squares)
- Parametric bootstrap hypothesis testing
- Comprehensive diagnostics and cross-checks
"""

__version__ = "1.0.0"
__author__ = "Research Software Engineering"

from rank1.config import Config, AnalysisConfig
from rank1.models.lowrank import LowRankModel, Rank1Model, Rank2Model
from rank1.models.fit import LowRankFitter
from rank1.models.bootstrap import BootstrapTester

__all__ = [
    "__version__",
    "Config",
    "AnalysisConfig",
    "LowRankModel",
    "Rank1Model",
    "Rank2Model",
    "LowRankFitter",
    "BootstrapTester",
]
