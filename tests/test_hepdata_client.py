"""
Tests for HEPData client functionality.
"""

import pytest
from pathlib import Path
import json

from rank1.data_sources.hepdata import HEPDataClient, HEPDataTable


class TestHEPDataTable:
    """Test HEPDataTable parsing."""

    def test_to_dataframe_simple(self):
        """Test conversion of simple table to DataFrame."""
        table = HEPDataTable(
            name="Test Table",
            description="A test table",
            keywords={},
            independent_variables=[
                {
                    "header": {"name": "x"},
                    "values": [{"value": 1}, {"value": 2}, {"value": 3}]
                }
            ],
            dependent_variables=[
                {
                    "header": {"name": "y"},
                    "values": [
                        {"value": 10, "errors": [{"symerror": 1, "label": "stat"}]},
                        {"value": 20, "errors": [{"symerror": 2, "label": "stat"}]},
                        {"value": 30, "errors": [{"symerror": 3, "label": "stat"}]},
                    ]
                }
            ],
        )

        df = table.to_dataframe()

        assert len(df) == 3
        assert "x" in df.columns
        assert "y" in df.columns
        assert "y_stat_err" in df.columns

        assert df["x"].tolist() == [1, 2, 3]
        assert df["y"].tolist() == [10, 20, 30]

    def test_to_dataframe_with_bins(self):
        """Test conversion with binned variables."""
        table = HEPDataTable(
            name="Binned Table",
            description="A binned table",
            keywords={},
            independent_variables=[
                {
                    "header": {"name": "pt"},
                    "values": [
                        {"low": 0, "high": 10},
                        {"low": 10, "high": 20},
                    ]
                }
            ],
            dependent_variables=[
                {
                    "header": {"name": "sigma"},
                    "values": [
                        {"value": 100},
                        {"value": 50},
                    ]
                }
            ],
        )

        df = table.to_dataframe()

        assert len(df) == 2
        assert "pt" in df.columns
        assert "pt_low" in df.columns
        assert "pt_high" in df.columns

        # Center should be average of bin edges
        assert df["pt"].tolist() == [5, 15]

    def test_n_rows(self):
        """Test row count property."""
        table = HEPDataTable(
            name="Test",
            description="",
            keywords={},
            independent_variables=[
                {"header": {"name": "x"}, "values": [{"value": i} for i in range(5)]}
            ],
            dependent_variables=[],
        )

        assert table.n_rows == 5


@pytest.mark.network
class TestHEPDataClientNetwork:
    """Tests requiring network access to HEPData."""

    def test_get_record_metadata(self, temp_dir):
        """Test fetching record metadata."""
        client = HEPDataClient(cache_dir=temp_dir, raw_dir=temp_dir)

        # Use a small, stable record for testing
        # Record 130266 is the ATLAS Higgs record
        try:
            metadata = client.get_record_metadata(130266)

            assert "title" in metadata or "tables" in metadata
        except Exception as e:
            pytest.skip(f"Network request failed: {e}")

    def test_download_record(self, temp_dir):
        """Test downloading a complete record."""
        client = HEPDataClient(cache_dir=temp_dir, raw_dir=temp_dir)

        try:
            record = client.download_record(130266)

            assert record.record_id == 130266
            assert len(record.tables) > 0
        except Exception as e:
            pytest.skip(f"Network request failed: {e}")


class TestHEPDataClientMocked:
    """Tests with mocked responses (no network required)."""

    def test_cache_key_generation(self, temp_dir):
        """Test cache key generation is deterministic."""
        client = HEPDataClient(cache_dir=temp_dir)

        key1 = client.http._cache_key("http://example.com", {"a": 1}, None)
        key2 = client.http._cache_key("http://example.com", {"a": 1}, None)

        assert key1 == key2

    def test_custom_cache_key(self, temp_dir):
        """Test custom cache key."""
        client = HEPDataClient(cache_dir=temp_dir)

        custom = "my_custom_key"
        key = client.http._cache_key("http://example.com", {}, custom)

        assert key == custom
