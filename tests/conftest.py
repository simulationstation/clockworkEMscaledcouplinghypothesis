"""
Pytest configuration and fixtures for rank-1 analysis tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_matrix_data():
    """Generate sample matrix data for testing."""
    np.random.seed(42)

    n_rows, n_cols = 5, 4

    # True rank-1 matrix
    u_true = np.array([1.0, 0.9, 1.1, 0.95, 1.05])
    v_true = np.array([1.0, 0.8, 0.6, 0.4])
    v_true /= np.linalg.norm(v_true)

    M_true = np.outer(u_true, v_true)

    # Add noise
    errors = 0.05 * np.abs(M_true) + 0.01
    noise = np.random.randn(n_rows, n_cols) * errors
    M_obs = M_true + noise

    # Create observation lists
    rows, cols = np.meshgrid(range(n_rows), range(n_cols), indexing='ij')
    row_idx = rows.ravel()
    col_idx = cols.ravel()
    values = M_obs.ravel()
    err = errors.ravel()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "values": values,
        "errors": err,
        "true_matrix": M_true,
        "observed_matrix": M_obs,
        "u_true": u_true,
        "v_true": v_true,
    }


@pytest.fixture
def sample_rank2_data():
    """Generate sample rank-2 matrix data for testing."""
    np.random.seed(123)

    n_rows, n_cols = 6, 5

    # True rank-2 matrix
    U = np.random.randn(n_rows, 2)
    V = np.random.randn(n_cols, 2)
    V, _ = np.linalg.qr(V)

    M_true = U @ V.T

    errors = 0.05 * np.abs(M_true) + 0.01
    noise = np.random.randn(n_rows, n_cols) * errors
    M_obs = M_true + noise

    rows, cols = np.meshgrid(range(n_rows), range(n_cols), indexing='ij')
    row_idx = rows.ravel()
    col_idx = cols.ravel()
    values = M_obs.ravel()
    err = errors.ravel()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "values": values,
        "errors": err,
        "true_matrix": M_true,
        "observed_matrix": M_obs,
    }


@pytest.fixture
def sparse_matrix_data():
    """Generate sparse matrix data with missing entries."""
    np.random.seed(456)

    n_rows, n_cols = 8, 6

    # True rank-1 matrix
    u = np.random.rand(n_rows) + 0.5
    v = np.random.rand(n_cols) + 0.5
    v /= np.linalg.norm(v)

    M_true = np.outer(u, v)

    # Random sparsity pattern (keep ~60% of entries)
    mask = np.random.rand(n_rows, n_cols) < 0.6

    # Observations
    row_idx = []
    col_idx = []
    values = []
    errors = []

    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                err = 0.05 * M_true[i, j] + 0.01
                val = M_true[i, j] + np.random.randn() * err
                row_idx.append(i)
                col_idx.append(j)
                values.append(val)
                errors.append(err)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "row_idx": np.array(row_idx),
        "col_idx": np.array(col_idx),
        "values": np.array(values),
        "errors": np.array(errors),
        "true_matrix": M_true,
        "mask": mask,
    }
