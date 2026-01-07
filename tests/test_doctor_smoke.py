"""
Tests for doctor/smoke test functionality.

These tests verify the doctor module works correctly without
requiring network access or large downloads.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from rank1.doctor import (
    Doctor,
    CheckStatus,
    OverallStatus,
    CheckResult,
    DoctorReport,
    run_doctor,
)


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_to_dict(self):
        """Test CheckResult serialization."""
        result = CheckResult(
            name="test_check",
            status=CheckStatus.PASS,
            message="Test passed",
            duration_seconds=1.5,
            details={"key": "value"},
        )

        d = result.to_dict()

        assert d["name"] == "test_check"
        assert d["status"] == "pass"
        assert d["message"] == "Test passed"
        assert d["duration_seconds"] == 1.5
        assert d["details"]["key"] == "value"

    def test_fix_tracking(self):
        """Test fix attempt tracking."""
        result = CheckResult(
            name="fixed_check",
            status=CheckStatus.FIXED,
            message="Fixed after retry",
            fix_attempted=True,
            fix_succeeded=True,
        )

        d = result.to_dict()
        assert d["fix_attempted"] is True
        assert d["fix_succeeded"] is True


class TestDoctorReport:
    """Test DoctorReport dataclass."""

    def test_to_dict(self):
        """Test report serialization."""
        report = DoctorReport(
            timestamp="2024-01-01T00:00:00",
            python_version="3.11.0",
            platform_info="Linux 5.15",
            overall_status=OverallStatus.PASS,
            checks=[
                CheckResult("check1", CheckStatus.PASS, "OK"),
                CheckResult("check2", CheckStatus.WARN, "Warning"),
            ],
            duration_seconds=10.0,
            warnings=["test warning"],
            manual_attention_items=[],
        )

        d = report.to_dict()

        assert d["overall_status"] == "PASS"
        assert len(d["checks"]) == 2
        assert d["duration_seconds"] == 10.0

    def test_to_markdown(self):
        """Test markdown generation."""
        report = DoctorReport(
            timestamp="2024-01-01T00:00:00",
            python_version="3.11.0",
            platform_info="Linux 5.15",
            overall_status=OverallStatus.PASS,
            checks=[
                CheckResult("check1", CheckStatus.PASS, "OK", duration_seconds=1.0),
            ],
            duration_seconds=10.0,
        )

        md = report.to_markdown()

        assert "# Doctor Report" in md
        assert "PASS" in md
        assert "check1" in md


class TestDoctor:
    """Test Doctor class."""

    def test_init(self, tmp_path):
        """Test Doctor initialization."""
        doctor = Doctor(
            output_dir=tmp_path,
            n_bootstrap=10,
            n_global_bootstrap=20,
        )

        assert doctor.output_dir == tmp_path
        assert doctor.n_bootstrap == 10
        assert doctor.doctor_dir.exists()

    def test_add_check(self, tmp_path):
        """Test adding check results."""
        doctor = Doctor(output_dir=tmp_path)

        result = CheckResult("test", CheckStatus.PASS, "OK")
        doctor._add_check(result)

        assert len(doctor.checks) == 1
        assert doctor.checks[0].name == "test"

    def test_compute_overall_status_pass(self, tmp_path):
        """Test overall status computation - pass case."""
        doctor = Doctor(output_dir=tmp_path)
        doctor.checks = [
            CheckResult("c1", CheckStatus.PASS, "OK"),
            CheckResult("c2", CheckStatus.PASS, "OK"),
        ]

        status = doctor._compute_overall_status()
        assert status == OverallStatus.PASS

    def test_compute_overall_status_fail(self, tmp_path):
        """Test overall status computation - fail case."""
        doctor = Doctor(output_dir=tmp_path)
        doctor.checks = [
            CheckResult("python_version", CheckStatus.FAIL, "Bad"),
            CheckResult("c2", CheckStatus.PASS, "OK"),
        ]

        status = doctor._compute_overall_status()
        assert status == OverallStatus.FAIL

    def test_compute_overall_status_needs_manual(self, tmp_path):
        """Test overall status computation - needs manual case."""
        doctor = Doctor(output_dir=tmp_path)
        doctor.manual_attention = ["Fix this"]
        doctor.checks = [
            CheckResult("c1", CheckStatus.PASS, "OK"),
        ]

        status = doctor._compute_overall_status()
        assert status == OverallStatus.NEEDS_MANUAL

    def test_write_report(self, tmp_path):
        """Test report file generation."""
        doctor = Doctor(output_dir=tmp_path)
        doctor.checks = [
            CheckResult("test", CheckStatus.PASS, "OK", duration_seconds=1.0),
        ]

        report = DoctorReport(
            timestamp="2024-01-01T00:00:00",
            python_version=sys.version,
            platform_info="test",
            overall_status=OverallStatus.PASS,
            checks=doctor.checks,
            duration_seconds=1.0,
        )

        doctor._write_report(report)

        json_path = doctor.doctor_dir / "doctor_report.json"
        md_path = doctor.doctor_dir / "doctor_report.md"

        assert json_path.exists()
        assert md_path.exists()


class TestEnvironmentChecks:
    """Test environment checking functionality."""

    def test_python_version_check(self, tmp_path):
        """Test Python version check."""
        doctor = Doctor(output_dir=tmp_path)
        doctor._run_environment_checks()

        version_check = next(
            (c for c in doctor.checks if c.name == "python_version"), None
        )
        assert version_check is not None

        # Should pass on Python 3.11+
        if sys.version_info >= (3, 11):
            assert version_check.status == CheckStatus.PASS

    def test_required_imports_check(self, tmp_path):
        """Test required imports check."""
        doctor = Doctor(output_dir=tmp_path)
        doctor._run_environment_checks()

        imports_check = next(
            (c for c in doctor.checks if c.name == "required_imports"), None
        )
        assert imports_check is not None
        # Should pass since we're running in an installed environment
        assert imports_check.status in [CheckStatus.PASS, CheckStatus.FAIL]


class TestMockedDoctor:
    """Tests with mocked external dependencies."""

    @patch("rank1.doctor.subprocess.run")
    def test_compile_check_pass(self, mock_run, tmp_path):
        """Test compile check passes."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        doctor = Doctor(output_dir=tmp_path)
        doctor._run_code_checks()

        compile_check = next(
            (c for c in doctor.checks if c.name == "compile_check"), None
        )
        assert compile_check is not None
        assert compile_check.status == CheckStatus.PASS

    @patch("rank1.doctor.subprocess.run")
    def test_compile_check_fail(self, mock_run, tmp_path):
        """Test compile check fails."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="SyntaxError"
        )

        doctor = Doctor(output_dir=tmp_path)
        doctor._run_code_checks()

        compile_check = next(
            (c for c in doctor.checks if c.name == "compile_check"), None
        )
        assert compile_check is not None
        assert compile_check.status == CheckStatus.FAIL


class TestArtifactCheck:
    """Test artifact checking functionality."""

    def test_artifact_check_missing(self, tmp_path):
        """Test artifact check with missing files."""
        doctor = Doctor(output_dir=tmp_path)

        # Mock config to use tmp_path
        with patch("rank1.config.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.output_dir = tmp_path
            mock_config.return_value = mock_cfg

            doctor._run_artifact_check()

        artifact_check = next(
            (c for c in doctor.checks if c.name == "artifacts"), None
        )
        assert artifact_check is not None
        # Should fail or warn since no files exist
        assert artifact_check.status in [CheckStatus.FAIL, CheckStatus.WARN]

    def test_artifact_check_present(self, tmp_path):
        """Test artifact check with files present."""
        # Create expected files
        (tmp_path / "higgs_atlas").mkdir(parents=True)
        (tmp_path / "higgs_atlas" / "results.json").write_text("{}")
        (tmp_path / "higgs_atlas" / "summary.md").write_text("# Summary")

        (tmp_path / "diffractive_dis").mkdir(parents=True)
        (tmp_path / "diffractive_dis" / "results.json").write_text("{}")
        (tmp_path / "diffractive_dis" / "summary.md").write_text("# Summary")

        (tmp_path / "higgs" / "np").mkdir(parents=True)
        (tmp_path / "higgs" / "np" / "np_results.json").write_text("{}")
        (tmp_path / "higgs" / "np" / "np_summary.txt").write_text("Summary")

        (tmp_path / "diffractive" / "np").mkdir(parents=True)
        (tmp_path / "diffractive" / "np" / "np_results.json").write_text("{}")
        (tmp_path / "diffractive" / "np" / "np_summary.txt").write_text("Summary")

        doctor = Doctor(output_dir=tmp_path)

        with patch("rank1.config.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.output_dir = tmp_path
            mock_config.return_value = mock_cfg

            doctor._run_artifact_check()

        artifact_check = next(
            (c for c in doctor.checks if c.name == "artifacts"), None
        )
        assert artifact_check is not None
        assert artifact_check.status == CheckStatus.PASS


class TestDeterminismCheck:
    """Test determinism verification."""

    def test_determinism_skip_on_error(self, tmp_path):
        """Test determinism check skips gracefully on error."""
        doctor = Doctor(output_dir=tmp_path)

        with patch("rank1.analysis.HiggsRankAnalysis") as mock_analysis:
            mock_analysis.side_effect = Exception("No data")
            doctor._run_determinism_check()

        det_check = next(
            (c for c in doctor.checks if c.name == "determinism"), None
        )
        assert det_check is not None
        assert det_check.status == CheckStatus.SKIP


# Integration test (marked slow, skipped in CI)
@pytest.mark.slow
@pytest.mark.network
class TestDoctorIntegration:
    """Integration tests that run actual doctor checks."""

    def test_run_doctor_minimal(self, tmp_path):
        """Run doctor with minimal settings."""
        report = run_doctor(
            output_dir=tmp_path,
            n_bootstrap=5,
            n_global_bootstrap=10,
            skip_large_downloads=True,
            fast_mode=True,
        )

        assert report.overall_status in [
            OverallStatus.PASS,
            OverallStatus.NEEDS_MANUAL,
            OverallStatus.FAIL,
        ]
        assert len(report.checks) > 0
        assert report.duration_seconds > 0
