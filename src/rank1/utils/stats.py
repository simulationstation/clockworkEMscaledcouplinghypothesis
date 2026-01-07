"""
Statistical utilities for rank-1 factorization analysis.
"""

import numpy as np
from scipy import stats
from scipy import linalg
from typing import Optional


def chi2_pvalue(chi2_stat: float, ndof: int) -> float:
    """
    Compute p-value from chi-squared statistic.

    Args:
        chi2_stat: Chi-squared test statistic
        ndof: Number of degrees of freedom

    Returns:
        p-value (probability of observing chi2 >= chi2_stat under null)
    """
    if ndof <= 0:
        return np.nan
    return 1.0 - stats.chi2.cdf(chi2_stat, ndof)


def clopper_pearson_ci(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper) confidence interval bounds
    """
    if n == 0:
        return (0.0, 1.0)

    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)

    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return (lower, upper)


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Compute weighted mean and its uncertainty.

    Args:
        values: Array of values
        weights: Array of weights (typically 1/variance)

    Returns:
        (mean, uncertainty) tuple
    """
    w = np.asarray(weights)
    v = np.asarray(values)

    if len(v) == 0:
        return (np.nan, np.nan)

    sum_w = np.sum(w)
    if sum_w == 0:
        return (np.nan, np.nan)

    mean = np.sum(w * v) / sum_w
    var = 1.0 / sum_w
    return (mean, np.sqrt(var))


def compute_correlation_matrix(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Args:
        covariance: Covariance matrix

    Returns:
        Correlation matrix
    """
    std = np.sqrt(np.diag(covariance))
    std_outer = np.outer(std, std)

    # Handle zero variances
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = covariance / std_outer
        corr = np.where(std_outer > 0, corr, 0)

    # Ensure diagonal is 1
    np.fill_diagonal(corr, 1.0)

    return corr


def covariance_to_correlation(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose covariance into standard deviations and correlation.

    Args:
        cov: Covariance matrix

    Returns:
        (std_devs, correlation_matrix) tuple
    """
    std = np.sqrt(np.diag(cov))
    corr = compute_correlation_matrix(cov)
    return (std, corr)


def cholesky_factor(cov: np.ndarray, regularize: float = 1e-10) -> np.ndarray:
    """
    Compute Cholesky factor with optional regularization.

    Args:
        cov: Positive semi-definite covariance matrix
        regularize: Small value to add to diagonal for stability

    Returns:
        Lower triangular Cholesky factor L where cov = L @ L.T
    """
    cov_reg = cov + regularize * np.eye(len(cov))
    try:
        L = linalg.cholesky(cov_reg, lower=True)
    except linalg.LinAlgError:
        # If still fails, try stronger regularization
        eigvals = linalg.eigvalsh(cov)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            cov_reg = cov + (abs(min_eig) + regularize) * np.eye(len(cov))
            L = linalg.cholesky(cov_reg, lower=True)
        else:
            raise

    return L


def whiten_residuals(
    residuals: np.ndarray,
    covariance: np.ndarray,
    regularize: float = 1e-10,
) -> np.ndarray:
    """
    Whiten residuals using covariance matrix.

    Args:
        residuals: Residual vector
        covariance: Covariance matrix
        regularize: Regularization for Cholesky

    Returns:
        Whitened residuals (should be ~N(0,I) if model is correct)
    """
    L = cholesky_factor(covariance, regularize)
    return linalg.solve_triangular(L, residuals, lower=True)


def compute_chi2(
    residuals: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
) -> float:
    """
    Compute chi-squared statistic.

    Args:
        residuals: (data - model) values
        uncertainties: Diagonal uncertainties (if no covariance)
        covariance: Full covariance matrix

    Returns:
        Chi-squared value
    """
    if covariance is not None:
        # Full covariance case
        whitened = whiten_residuals(residuals, covariance)
        return float(np.sum(whitened**2))
    elif uncertainties is not None:
        # Diagonal case
        return float(np.sum((residuals / uncertainties) ** 2))
    else:
        raise ValueError("Must provide either uncertainties or covariance")


def singular_value_spectrum(
    matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute singular values of a (possibly weighted) matrix.

    Args:
        matrix: 2D matrix
        weights: Optional weight matrix (same shape as matrix)

    Returns:
        Array of singular values in descending order
    """
    if weights is not None:
        # Weight the matrix elementwise
        weighted = matrix * np.sqrt(weights)
    else:
        weighted = matrix

    # Handle NaN/missing values
    weighted = np.nan_to_num(weighted, nan=0.0)

    _, s, _ = linalg.svd(weighted, full_matrices=False)
    return s


def explained_variance_ratio(singular_values: np.ndarray, rank: int) -> float:
    """
    Compute fraction of variance explained by top-k singular values.

    Args:
        singular_values: Singular values in descending order
        rank: Number of top singular values to consider

    Returns:
        Fraction of total variance explained (0 to 1)
    """
    total = np.sum(singular_values**2)
    if total == 0:
        return 1.0

    explained = np.sum(singular_values[:rank] ** 2)
    return float(explained / total)


def condition_number(matrix: np.ndarray, regularize: float = 1e-10) -> float:
    """
    Compute condition number of a matrix.

    Args:
        matrix: 2D matrix
        regularize: Regularization to avoid division by zero

    Returns:
        Condition number (ratio of largest to smallest singular value)
    """
    s = linalg.svdvals(matrix)
    if len(s) == 0:
        return np.inf

    max_s = np.max(s)
    min_s = np.min(s)

    if min_s < regularize:
        return max_s / regularize

    return float(max_s / min_s)


def bootstrap_pvalue(
    observed_statistic: float,
    bootstrap_statistics: np.ndarray,
    alternative: str = "greater",
) -> tuple[float, tuple[float, float]]:
    """
    Compute bootstrap p-value with Clopper-Pearson confidence interval.

    Uses the (k+1)/(B+1) estimator for proper calibration.

    Args:
        observed_statistic: Observed test statistic
        bootstrap_statistics: Array of bootstrap statistics under null
        alternative: 'greater' (one-sided) or 'two-sided'

    Returns:
        (p_value, (ci_lower, ci_upper)) tuple
    """
    B = len(bootstrap_statistics)

    if alternative == "greater":
        k = np.sum(bootstrap_statistics >= observed_statistic)
    elif alternative == "two-sided":
        # Two-sided: count extreme values on both sides
        k = np.sum(np.abs(bootstrap_statistics) >= np.abs(observed_statistic))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # (k+1)/(B+1) estimator
    p_value = (k + 1) / (B + 1)

    # Clopper-Pearson CI for the p-value
    ci = clopper_pearson_ci(k, B, alpha=0.05)

    return (p_value, ci)


def ks_test_uniformity(pvalues: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test for uniformity of p-values.

    Useful for validating bootstrap calibration.

    Args:
        pvalues: Array of p-values that should be uniform under null

    Returns:
        (KS statistic, p-value)
    """
    result = stats.kstest(pvalues, "uniform")
    return (float(result.statistic), float(result.pvalue))
