"""
Dataset identity verification with cell spot-checks.

Verifies that loaded data matches expected values from published sources.
This catches:
- Wrong table loaded from HEPData
- Data corruption during extraction
- Incorrect unit conversions
- Silent fallback to wrong data

Each dataset has spot-check values manually extracted from publications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json

import numpy as np

from rank1.logging import get_logger
from rank1.datasets.base import MatrixData

logger = get_logger()


@dataclass
class SpotCheck:
    """A single spot-check verification."""
    description: str
    row_label: str
    col_label: str
    expected_value: float
    expected_error: float
    tolerance: float = 0.05  # 5% relative tolerance
    source_reference: str = ""


@dataclass
class SpotCheckResult:
    """Result of a spot-check verification."""
    check: SpotCheck
    passed: bool
    actual_value: Optional[float]
    actual_error: Optional[float]
    relative_diff: Optional[float]
    message: str


@dataclass
class IdentityReport:
    """Complete dataset identity verification report."""
    dataset_name: str
    all_passed: bool
    n_checks: int
    n_passed: int
    results: list[SpotCheckResult]
    provenance_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "all_passed": self.all_passed,
            "n_checks": self.n_checks,
            "n_passed": self.n_passed,
            "results": [
                {
                    "description": r.check.description,
                    "row_label": r.check.row_label,
                    "col_label": r.check.col_label,
                    "expected_value": r.check.expected_value,
                    "actual_value": r.actual_value,
                    "passed": r.passed,
                    "relative_diff": r.relative_diff,
                    "message": r.message,
                }
                for r in self.results
            ],
            "provenance_warnings": self.provenance_warnings,
        }


# ============================================================================
# SPOT-CHECK DEFINITIONS
# These are manually extracted from published papers/HEPData
# ============================================================================

HIGGS_SPOT_CHECKS = [
    # From HEPData record 130266 - ATLAS Higgs combined measurements
    # Values from published paper / HEPData tables
    SpotCheck(
        description="ggF → γγ signal strength",
        row_label="ggF",
        col_label="γγ",
        expected_value=1.04,
        expected_error=0.10,
        tolerance=0.15,  # Allow some tolerance for extraction
        source_reference="HEPData:130266, Table 2",
    ),
    SpotCheck(
        description="VBF → ττ signal strength",
        row_label="VBF",
        col_label="ττ",
        expected_value=1.05,
        expected_error=0.12,
        tolerance=0.15,
        source_reference="HEPData:130266",
    ),
    SpotCheck(
        description="ttH → γγ signal strength (distinctive)",
        row_label="ttH",
        col_label="γγ",
        expected_value=1.38,  # ttH is notably enhanced
        expected_error=0.25,
        tolerance=0.20,
        source_reference="HEPData:130266",
    ),
]

ELASTIC_SPOT_CHECKS = [
    # From TOTEM published data
    # 7 TeV: EPL 95 (2011) 41001
    SpotCheck(
        description="7 TeV dσ/dt at |t|=0.05 GeV²",
        row_label="7 TeV",
        col_label="|t|=0.050",  # Approximate match
        expected_value=315.0,  # mb/GeV² (before log transform)
        expected_error=7.0,
        tolerance=0.10,
        source_reference="TOTEM EPL 95 (2011) 41001",
    ),
    SpotCheck(
        description="7 TeV dσ/dt at |t|=0.10 GeV²",
        row_label="7 TeV",
        col_label="|t|=0.100",
        expected_value=91.0,  # mb/GeV²
        expected_error=2.0,
        tolerance=0.10,
        source_reference="TOTEM EPL 95 (2011) 41001",
    ),
]

DIFFRACTIVE_SPOT_CHECKS = [
    # From H1/ZEUS diffractive structure function measurements
    # These are harder to check exactly due to variable binning
    SpotCheck(
        description="Typical σ_r^D at Q²=10, β=0.1, xP=0.003",
        row_label="Q2=10.0_beta=0.100",
        col_label="xP=0.00300",
        expected_value=0.05,  # Order of magnitude check
        expected_error=0.01,
        tolerance=0.50,  # Loose tolerance for binning differences
        source_reference="H1: HEPData ins718189",
    ),
]


class DatasetIdentityVerifier:
    """
    Verify dataset identity using spot-checks.

    Compares loaded data against known reference values to ensure
    the correct data has been loaded.
    """

    def __init__(self, matrix_data: MatrixData):
        self.data = matrix_data
        self._value_dict: dict[tuple[str, str], tuple[float, float]] = {}
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Build lookup dictionary for fast access."""
        for obs in self.data.observations:
            key = (obs.row_label, obs.col_label)
            self._value_dict[key] = (obs.value, obs.total_err)

    def _find_closest_match(
        self, row_pattern: str, col_pattern: str
    ) -> Optional[tuple[str, str]]:
        """Find closest matching cell for patterns."""
        # Try exact match first
        for key in self._value_dict.keys():
            if row_pattern in key[0] and col_pattern in key[1]:
                return key

        # Try partial match
        for key in self._value_dict.keys():
            row_match = any(
                p in key[0] for p in row_pattern.split("_")
            )
            col_match = any(
                p in key[1] for p in col_pattern.split("=")
            )
            if row_match and col_match:
                return key

        return None

    def run_spot_check(self, check: SpotCheck) -> SpotCheckResult:
        """Run a single spot-check."""
        # Try exact match
        key = (check.row_label, check.col_label)
        if key not in self._value_dict:
            # Try pattern matching
            matched_key = self._find_closest_match(check.row_label, check.col_label)
            if matched_key:
                key = matched_key
            else:
                return SpotCheckResult(
                    check=check,
                    passed=False,
                    actual_value=None,
                    actual_error=None,
                    relative_diff=None,
                    message=f"Cell not found: ({check.row_label}, {check.col_label})",
                )

        actual_value, actual_error = self._value_dict[key]

        # Handle log-transformed values
        # If actual value is negative, it's log-transformed
        if actual_value < 0:
            # Transform expected value to log scale for comparison
            if check.expected_value > 0:
                expected_log = np.log(check.expected_value)
                relative_diff = abs(actual_value - expected_log) / abs(expected_log)
                # More lenient check for log-transformed
                passed = relative_diff < 0.5
                message = f"Log-scale comparison: {actual_value:.3f} vs {expected_log:.3f}"
            else:
                passed = False
                relative_diff = None
                message = "Cannot compare log-transformed with non-positive expected"
        else:
            # Direct comparison
            relative_diff = abs(actual_value - check.expected_value) / abs(check.expected_value)
            passed = relative_diff <= check.tolerance
            message = "" if passed else f"Value mismatch: {actual_value:.4f} vs {check.expected_value:.4f}"

        return SpotCheckResult(
            check=check,
            passed=passed,
            actual_value=actual_value,
            actual_error=actual_error,
            relative_diff=relative_diff,
            message=message,
        )

    def verify(self, spot_checks: list[SpotCheck]) -> IdentityReport:
        """Run all spot-checks and generate report."""
        results = []
        provenance_warnings = []

        # Check provenance for warnings
        if self.data.provenance:
            if self.data.provenance.has_fallbacks:
                for fb in self.data.provenance.fallbacks:
                    provenance_warnings.append(
                        f"FALLBACK at {fb.step}: {fb.reason} -> {fb.fallback_to}"
                    )
            if self.data.provenance.is_placeholder:
                provenance_warnings.append(
                    "DATA IS PLACEHOLDER - spot-checks will fail"
                )

        # Run spot-checks
        for check in spot_checks:
            result = self.run_spot_check(check)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(f"  [{status}] {check.description}")
            if not result.passed:
                logger.warning(f"    {result.message}")

        n_passed = sum(1 for r in results if r.passed)
        all_passed = n_passed == len(results) and len(provenance_warnings) == 0

        return IdentityReport(
            dataset_name=self.data.name,
            all_passed=all_passed,
            n_checks=len(results),
            n_passed=n_passed,
            results=results,
            provenance_warnings=provenance_warnings,
        )


def verify_higgs_identity(matrix_data: MatrixData) -> IdentityReport:
    """Verify ATLAS Higgs dataset identity."""
    logger.info("Verifying ATLAS Higgs dataset identity...")
    verifier = DatasetIdentityVerifier(matrix_data)
    return verifier.verify(HIGGS_SPOT_CHECKS)


def verify_elastic_identity(matrix_data: MatrixData) -> IdentityReport:
    """Verify TOTEM elastic dataset identity."""
    logger.info("Verifying TOTEM elastic dataset identity...")
    verifier = DatasetIdentityVerifier(matrix_data)
    return verifier.verify(ELASTIC_SPOT_CHECKS)


def verify_diffractive_identity(matrix_data: MatrixData) -> IdentityReport:
    """Verify H1/ZEUS diffractive dataset identity."""
    logger.info("Verifying H1/ZEUS diffractive dataset identity...")
    verifier = DatasetIdentityVerifier(matrix_data)
    return verifier.verify(DIFFRACTIVE_SPOT_CHECKS)


def verify_all_datasets(
    higgs_data: Optional[MatrixData] = None,
    elastic_data: Optional[MatrixData] = None,
    diffractive_data: Optional[MatrixData] = None,
) -> dict[str, IdentityReport]:
    """Verify identity of all provided datasets."""
    reports = {}

    if higgs_data:
        reports["higgs"] = verify_higgs_identity(higgs_data)

    if elastic_data:
        reports["elastic"] = verify_elastic_identity(elastic_data)

    if diffractive_data:
        reports["diffractive"] = verify_diffractive_identity(diffractive_data)

    return reports
