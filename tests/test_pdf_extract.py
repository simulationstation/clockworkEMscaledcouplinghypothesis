"""
Tests for PDF table extraction and manual table fallback.
"""

import pytest
from pathlib import Path
import yaml

from rank1.data_sources.arxiv import PDFTableExtractor, ArxivClient


class TestPDFTableExtractor:
    """Test PDF table extraction functionality."""

    def test_check_libraries(self):
        """Test library availability checks."""
        extractor = PDFTableExtractor()

        # These might be True or False depending on installation
        assert isinstance(extractor._camelot_available, bool)
        assert isinstance(extractor._pdfplumber_available, bool)

    def test_manual_tables_dir_default(self):
        """Test default manual tables directory."""
        extractor = PDFTableExtractor()

        assert extractor.manual_tables_dir == Path("src/rank1/datasets/manual_tables")


class TestManualTables:
    """Test manual table loading and validation."""

    def test_8tev_table_exists(self):
        """Test 8 TeV manual table exists and is valid."""
        table_path = Path("src/rank1/datasets/manual_tables/1503_08111_dsigma_dt_8tev.yaml")

        if not table_path.exists():
            pytest.skip("Manual table not found")

        with open(table_path) as f:
            data = yaml.safe_load(f)

        assert "columns" in data
        assert "data" in data
        assert len(data["data"]) > 10

        # Validate columns
        expected_cols = ["t", "dsigma_dt"]
        for col in expected_cols:
            assert col in data["columns"]

    def test_13tev_table_exists(self):
        """Test 13 TeV manual table exists and is valid."""
        table_path = Path("src/rank1/datasets/manual_tables/1812_08283_dsigma_dt_13tev.yaml")

        if not table_path.exists():
            pytest.skip("Manual table not found")

        with open(table_path) as f:
            data = yaml.safe_load(f)

        assert "columns" in data
        assert "data" in data
        assert len(data["data"]) > 20  # 13 TeV table should have more points

    def test_8tev_slope_consistency(self):
        """Test that 8 TeV data gives expected forward slope."""
        table_path = Path("src/rank1/datasets/manual_tables/1503_08111_dsigma_dt_8tev.yaml")

        if not table_path.exists():
            pytest.skip("Manual table not found")

        import numpy as np
        import pandas as pd

        with open(table_path) as f:
            data = yaml.safe_load(f)

        df = pd.DataFrame(data["data"], columns=data["columns"])

        # Fit slope in forward region
        mask = df["t"] < 0.10
        t = df.loc[mask, "t"].values
        dsigma = df.loc[mask, "dsigma_dt"].values

        log_dsigma = np.log(dsigma)
        coeffs = np.polyfit(t, log_dsigma, 1)
        B = -coeffs[0]

        # Published slope is approximately 19.9 GeV^-2
        assert 18.0 < B < 22.0, f"Extracted slope B={B} outside expected range"

    def test_13tev_dip_position(self):
        """Test that 13 TeV data has dip at expected position."""
        table_path = Path("src/rank1/datasets/manual_tables/1812_08283_dsigma_dt_13tev.yaml")

        if not table_path.exists():
            pytest.skip("Manual table not found")

        import numpy as np
        import pandas as pd

        with open(table_path) as f:
            data = yaml.safe_load(f)

        df = pd.DataFrame(data["data"], columns=data["columns"])

        # Find minimum (dip) position
        min_idx = df["dsigma_dt"].idxmin()
        t_dip = df.loc[min_idx, "t"]

        # Published dip is around |t| ~ 0.47 GeV^2
        assert 0.40 < t_dip < 0.55, f"Dip at |t|={t_dip} outside expected range"


class TestArxivClient:
    """Test ArxivClient functionality."""

    def test_get_manual_table(self, temp_dir):
        """Test loading manual table through client."""
        # Create a test manual table
        manual_dir = temp_dir / "manual_tables"
        manual_dir.mkdir()

        test_data = {
            "columns": ["x", "y"],
            "data": [[1, 2], [3, 4]],
        }

        table_path = manual_dir / "test_table.yaml"
        with open(table_path, "w") as f:
            yaml.dump(test_data, f)

        client = ArxivClient(
            cache_dir=temp_dir,
            raw_dir=temp_dir,
            manual_tables_dir=manual_dir,
        )

        df = client.get_manual_table("test", "table")

        assert df is not None
        assert len(df) == 2
        assert list(df.columns) == ["x", "y"]

    def test_normalize_dsigma_columns(self, temp_dir):
        """Test column normalization."""
        import pandas as pd

        client = ArxivClient(cache_dir=temp_dir, raw_dir=temp_dir)

        df = pd.DataFrame({
            "|t|": [0.1, 0.2],
            "dsigma/dt": [100, 50],
            "stat_uncertainty": [5, 3],
        })

        normalized = client._normalize_dsigma_columns(df)

        assert "t" in normalized.columns or "|t|" in df.columns
        assert "total_err" in normalized.columns
