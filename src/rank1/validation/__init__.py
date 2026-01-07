"""
Data validation and verification utilities.

This module provides tools for:
- Dataset identity verification (spot-checks against published values)
- Statistical calibration (null uniformity, power curves)
- Robustness analysis (sensitivity to analysis choices)
"""

from rank1.validation.identity import (
    DatasetIdentityVerifier,
    SpotCheck,
    IdentityReport,
    verify_all_datasets,
)
from rank1.validation.calibration import (
    NullCalibrator,
    PowerAnalyzer,
    CalibrationReport,
)

__all__ = [
    "DatasetIdentityVerifier",
    "SpotCheck",
    "IdentityReport",
    "verify_all_datasets",
    "NullCalibrator",
    "PowerAnalyzer",
    "CalibrationReport",
]
