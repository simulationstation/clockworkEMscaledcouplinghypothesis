"""Reporting utilities for generating figures, tables, and markdown reports."""

from rank1.reporting.figures import FigureGenerator
from rank1.reporting.tables import TableGenerator
from rank1.reporting.report_md import ReportGenerator

# NP (New Physics Sensitive) reporting
from rank1.reporting.np_report import NPReportGenerator
from rank1.reporting.np_figures import NPFigureGenerator

__all__ = [
    "FigureGenerator",
    "TableGenerator",
    "ReportGenerator",
    # NP reporting
    "NPReportGenerator",
    "NPFigureGenerator",
]
