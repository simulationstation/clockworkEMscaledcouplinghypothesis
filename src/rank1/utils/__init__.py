"""Utility modules for the rank-1 analysis pipeline."""

from rank1.utils.http import HTTPClient, download_file
from rank1.utils.cache import Cache, cached
from rank1.utils.hashing import compute_checksum, verify_checksum
from rank1.utils.parallel import parallel_map, get_executor
from rank1.utils.stats import (
    chi2_pvalue,
    clopper_pearson_ci,
    weighted_mean,
    compute_correlation_matrix,
)

__all__ = [
    "HTTPClient",
    "download_file",
    "Cache",
    "cached",
    "compute_checksum",
    "verify_checksum",
    "parallel_map",
    "get_executor",
    "chi2_pvalue",
    "clopper_pearson_ci",
    "weighted_mean",
    "compute_correlation_matrix",
]
