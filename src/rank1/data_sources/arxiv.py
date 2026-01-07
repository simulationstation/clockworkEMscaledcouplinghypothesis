"""
arXiv client for fetching PDFs and extracting tables.

Supports:
- PDF download from arXiv
- Table extraction using camelot/pdfplumber
- Fallback to manual YAML tables with validation
"""

import re
import yaml
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

from rank1.utils.http import HTTPClient
from rank1.logging import get_logger

logger = get_logger()

ARXIV_PDF_URL = "https://arxiv.org/pdf"
ARXIV_ABS_URL = "https://arxiv.org/abs"


@dataclass
class ExtractedTable:
    """A table extracted from a PDF."""

    page: int
    table_index: int
    dataframe: pd.DataFrame
    extraction_method: str
    confidence: float

    @property
    def n_rows(self) -> int:
        return len(self.dataframe)

    @property
    def n_cols(self) -> int:
        return len(self.dataframe.columns)


class PDFTableExtractor:
    """
    Extract tables from PDFs using multiple methods.

    Tries camelot first (best for structured tables), then pdfplumber.
    Falls back to manual YAML files if extraction fails.
    """

    def __init__(self, manual_tables_dir: Optional[Path] = None):
        """
        Initialize extractor.

        Args:
            manual_tables_dir: Directory containing manual YAML table files
        """
        self.manual_tables_dir = manual_tables_dir or Path(
            "src/rank1/datasets/manual_tables"
        )
        self._camelot_available = self._check_camelot()
        self._pdfplumber_available = self._check_pdfplumber()

    def _check_camelot(self) -> bool:
        """Check if camelot is available."""
        try:
            import camelot

            return True
        except ImportError:
            logger.debug("camelot not available")
            return False

    def _check_pdfplumber(self) -> bool:
        """Check if pdfplumber is available."""
        try:
            import pdfplumber

            return True
        except ImportError:
            logger.debug("pdfplumber not available")
            return False

    def extract_tables(
        self,
        pdf_path: Path,
        pages: Optional[str] = None,
    ) -> list[ExtractedTable]:
        """
        Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Page specification (e.g., '1-3', 'all')

        Returns:
            List of ExtractedTable objects
        """
        tables = []

        # Try camelot first
        if self._camelot_available:
            try:
                camelot_tables = self._extract_with_camelot(pdf_path, pages)
                tables.extend(camelot_tables)
            except Exception as e:
                logger.warning(f"Camelot extraction failed: {e}")

        # If camelot failed or found nothing, try pdfplumber
        if not tables and self._pdfplumber_available:
            try:
                pdfplumber_tables = self._extract_with_pdfplumber(pdf_path, pages)
                tables.extend(pdfplumber_tables)
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")

        return tables

    def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: Optional[str] = None,
    ) -> list[ExtractedTable]:
        """Extract tables using camelot."""
        import camelot

        if pages is None:
            pages = "all"

        logger.debug(f"Extracting tables with camelot from {pdf_path}")

        # Try lattice mode first (for tables with lines)
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor="lattice")
        except Exception:
            # Fall back to stream mode
            tables = camelot.read_pdf(str(pdf_path), pages=pages, flavor="stream")

        results = []
        for i, table in enumerate(tables):
            df = table.df

            # Skip empty or very small tables
            if len(df) < 2 or len(df.columns) < 2:
                continue

            results.append(
                ExtractedTable(
                    page=table.page,
                    table_index=i,
                    dataframe=df,
                    extraction_method="camelot",
                    confidence=table.accuracy if hasattr(table, "accuracy") else 0.0,
                )
            )

        logger.info(f"Camelot extracted {len(results)} tables")
        return results

    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[str] = None,
    ) -> list[ExtractedTable]:
        """Extract tables using pdfplumber."""
        import pdfplumber

        logger.debug(f"Extracting tables with pdfplumber from {pdf_path}")

        results = []

        with pdfplumber.open(pdf_path) as pdf:
            page_nums = self._parse_pages(pages, len(pdf.pages))

            for page_num in page_nums:
                page = pdf.pages[page_num]
                tables = page.extract_tables()

                for i, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])

                    if len(df.columns) < 2:
                        continue

                    results.append(
                        ExtractedTable(
                            page=page_num + 1,
                            table_index=i,
                            dataframe=df,
                            extraction_method="pdfplumber",
                            confidence=0.5,  # pdfplumber doesn't provide confidence
                        )
                    )

        logger.info(f"pdfplumber extracted {len(results)} tables")
        return results

    def _parse_pages(self, pages: Optional[str], total: int) -> list[int]:
        """Parse page specification to list of 0-indexed page numbers."""
        if pages is None or pages == "all":
            return list(range(total))

        result = []
        for part in pages.split(","):
            if "-" in part:
                start, end = part.split("-")
                result.extend(range(int(start) - 1, int(end)))
            else:
                result.append(int(part) - 1)

        return [p for p in result if 0 <= p < total]


class ArxivClient:
    """
    Client for downloading arXiv PDFs and extracting data.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        manual_tables_dir: Optional[Path] = None,
    ):
        """
        Initialize arXiv client.

        Args:
            cache_dir: Directory for HTTP cache
            raw_dir: Directory for downloaded PDFs
            manual_tables_dir: Directory for manual YAML tables
        """
        self.cache_dir = cache_dir or Path("data/cache")
        self.raw_dir = raw_dir or Path("data/raw/arxiv")
        self.manual_tables_dir = manual_tables_dir or Path(
            "src/rank1/datasets/manual_tables"
        )
        self.http = HTTPClient(cache_dir=self.cache_dir)
        self.extractor = PDFTableExtractor(self.manual_tables_dir)

    def download_pdf(
        self,
        arxiv_id: str,
        force: bool = False,
    ) -> Path:
        """
        Download PDF from arXiv.

        Args:
            arxiv_id: arXiv identifier (e.g., '1503.08111')
            force: Force re-download

        Returns:
            Path to downloaded PDF
        """
        # Clean up arxiv ID
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")

        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        dest = self.raw_dir / filename

        if not force and dest.exists():
            logger.debug(f"Using cached PDF: {dest}")
            return dest

        url = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"
        return self.http.download(url, dest, force=force)

    def extract_tables(
        self,
        arxiv_id: str,
        pages: Optional[str] = None,
        force_redownload: bool = False,
    ) -> list[ExtractedTable]:
        """
        Download PDF and extract tables.

        Args:
            arxiv_id: arXiv identifier
            pages: Page specification
            force_redownload: Force PDF re-download

        Returns:
            List of ExtractedTable objects
        """
        pdf_path = self.download_pdf(arxiv_id, force=force_redownload)
        return self.extractor.extract_tables(pdf_path, pages)

    def get_manual_table(
        self,
        arxiv_id: str,
        table_name: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get manually curated table from YAML file.

        Args:
            arxiv_id: arXiv identifier
            table_name: Table name/identifier

        Returns:
            DataFrame or None if not found
        """
        arxiv_clean = arxiv_id.replace(".", "_").replace("/", "_")
        yaml_path = self.manual_tables_dir / f"{arxiv_clean}_{table_name}.yaml"

        if not yaml_path.exists():
            logger.debug(f"Manual table not found: {yaml_path}")
            return None

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if "data" in data and "columns" in data:
            return pd.DataFrame(data["data"], columns=data["columns"])
        elif "data" in data:
            return pd.DataFrame(data["data"])
        elif "columns" in data and "values" in data:
            return pd.DataFrame(data["values"], columns=data["columns"])
        else:
            # Assume it's directly a list of dicts
            return pd.DataFrame(data)

    def load_dsigma_dt_table(
        self,
        arxiv_id: str,
        energy_tev: float,
    ) -> Optional[pd.DataFrame]:
        """
        Load dσ/dt table for a specific energy.

        This is a high-level method that tries extraction then falls back
        to manual tables.

        Args:
            arxiv_id: arXiv identifier
            energy_tev: Center-of-mass energy in TeV

        Returns:
            DataFrame with columns: t, dsigma_dt, stat_err, sys_err, total_err
        """
        # First try manual table
        table_name = f"dsigma_dt_{int(energy_tev)}tev"
        df = self.get_manual_table(arxiv_id, table_name)

        if df is not None:
            logger.info(f"Using manual table for {arxiv_id} at {energy_tev} TeV")
            return self._normalize_dsigma_columns(df)

        # Try PDF extraction
        logger.info(f"Attempting PDF extraction for {arxiv_id}")
        tables = self.extract_tables(arxiv_id)

        # Find the most likely dσ/dt table
        for table in tables:
            if self._looks_like_dsigma_table(table.dataframe):
                logger.info(
                    f"Found dσ/dt table in {arxiv_id} "
                    f"(page {table.page}, {table.n_rows} rows)"
                )
                return self._normalize_dsigma_columns(table.dataframe)

        logger.warning(f"Could not find dσ/dt table for {arxiv_id}")
        return None

    def _looks_like_dsigma_table(self, df: pd.DataFrame) -> bool:
        """Check if a DataFrame looks like a dσ/dt table."""
        # Look for |t| or -t column and dσ/dt-like column
        col_str = " ".join(str(c).lower() for c in df.columns)
        has_t = "t" in col_str or "|t|" in col_str
        has_dsigma = "dsigma" in col_str or "sigma" in col_str or "mb" in col_str

        # Also check data types - should have numeric values
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            has_numeric = len(numeric_cols) >= 2
        except Exception:
            has_numeric = False

        return has_t or has_dsigma or has_numeric

    def _normalize_dsigma_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names for dσ/dt table.

        Ensures consistent column naming:
        - t or t_abs: momentum transfer |t| in GeV²
        - dsigma_dt: differential cross section in mb/GeV²
        - stat_err, sys_err, total_err: uncertainties
        """
        df = df.copy()

        # Column mapping
        column_map = {}
        for col in df.columns:
            col_lower = str(col).lower()

            if "|t|" in col_lower or "t_abs" in col_lower or col_lower == "t":
                column_map[col] = "t"
            elif "dsigma" in col_lower or "dsig" in col_lower:
                if "stat" in col_lower:
                    column_map[col] = "stat_err"
                elif "sys" in col_lower:
                    column_map[col] = "sys_err"
                elif "err" in col_lower or "unc" in col_lower:
                    column_map[col] = "total_err"
                else:
                    column_map[col] = "dsigma_dt"
            elif "stat" in col_lower and ("err" in col_lower or "unc" in col_lower):
                column_map[col] = "stat_err"
            elif "sys" in col_lower and ("err" in col_lower or "unc" in col_lower):
                column_map[col] = "sys_err"
            elif "err" in col_lower or "unc" in col_lower:
                column_map[col] = "total_err"

        df = df.rename(columns=column_map)

        # Ensure required columns exist
        required = ["t", "dsigma_dt"]
        for col in required:
            if col not in df.columns:
                # Try to infer from position
                if col == "t" and len(df.columns) >= 1:
                    df = df.rename(columns={df.columns[0]: "t"})
                elif col == "dsigma_dt" and len(df.columns) >= 2:
                    df = df.rename(columns={df.columns[1]: "dsigma_dt"})

        # Ensure error columns
        if "stat_err" not in df.columns:
            df["stat_err"] = 0.0
        if "sys_err" not in df.columns:
            df["sys_err"] = 0.0
        if "total_err" not in df.columns:
            if "stat_err" in df.columns and "sys_err" in df.columns:
                df["total_err"] = np.sqrt(
                    df["stat_err"].astype(float) ** 2 + df["sys_err"].astype(float) ** 2
                )
            else:
                df["total_err"] = 0.0

        # Convert to numeric
        for col in ["t", "dsigma_dt", "stat_err", "sys_err", "total_err"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


def fetch_totem_8tev_dsigma(
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch TOTEM 8 TeV dσ/dt data.

    Source: arXiv:1503.08111

    Returns:
        DataFrame with dσ/dt data
    """
    client = ArxivClient(cache_dir=cache_dir, raw_dir=raw_dir)
    df = client.load_dsigma_dt_table("1503.08111", 8.0)

    if df is None:
        # Return manual table as fallback
        df = client.get_manual_table("1503.08111", "dsigma_dt_8tev")

    return df


def fetch_totem_13tev_dsigma(
    cache_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch TOTEM 13 TeV dσ/dt data.

    Source: arXiv:1812.08283

    Returns:
        DataFrame with dσ/dt data
    """
    client = ArxivClient(cache_dir=cache_dir, raw_dir=raw_dir)
    df = client.load_dsigma_dt_table("1812.08283", 13.0)

    if df is None:
        # Return manual table as fallback
        df = client.get_manual_table("1812.08283", "dsigma_dt_13tev")

    return df
