"""
Low-rank matrix models for factorization tests.

Implements rank-1 and rank-2 matrix decompositions with gauge fixing
to avoid scale degeneracy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import linalg


@dataclass
class LowRankModel(ABC):
    """
    Abstract base class for low-rank matrix models.

    A low-rank model approximates a matrix M as a sum of outer products:
    M ≈ Σ_k u_k ⊗ v_k

    For rank-1: M_ij ≈ u_i * v_j
    For rank-2: M_ij ≈ u1_i*v1_j + u2_i*v2_j
    """

    n_rows: int
    n_cols: int
    rank: int

    @abstractmethod
    def predict(self, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
        """
        Predict values at given (row, col) indices.

        Args:
            row_idx: Row indices
            col_idx: Column indices

        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def predict_matrix(self) -> np.ndarray:
        """
        Predict full matrix.

        Returns:
            Predicted matrix of shape (n_rows, n_cols)
        """
        pass

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Get flattened parameter vector."""
        pass

    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from flattened vector."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of free parameters."""
        pass

    def jacobian(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Jacobian of predictions w.r.t. parameters.

        Default implementation uses numerical differentiation.
        Override for analytic Jacobian.

        Args:
            row_idx: Row indices
            col_idx: Column indices

        Returns:
            Jacobian matrix of shape (n_obs, n_params)
        """
        eps = 1e-7
        n_obs = len(row_idx)
        params = self.get_params()
        jac = np.zeros((n_obs, self.n_params))

        for i in range(self.n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            self.set_params(params_plus)
            pred_plus = self.predict(row_idx, col_idx)

            params_minus = params.copy()
            params_minus[i] -= eps
            self.set_params(params_minus)
            pred_minus = self.predict(row_idx, col_idx)

            jac[:, i] = (pred_plus - pred_minus) / (2 * eps)

        self.set_params(params)
        return jac


@dataclass
class Rank1Model(LowRankModel):
    """
    Rank-1 matrix model: M_ij = u_i * v_j

    Gauge fixing: ||v||_2 = 1 (unit norm for v, scale absorbed into u)

    Parameters:
    - u: row factors (length n_rows)
    - v: column factors (length n_cols - 1, last component is derived)
    """

    rank: int = field(default=1, init=False)

    # Model parameters
    u: np.ndarray = field(default=None, repr=False)
    v: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.u is None:
            self.u = np.ones(self.n_rows)
        if self.v is None:
            self.v = np.ones(self.n_cols)
            self.v /= np.linalg.norm(self.v)

    def predict(self, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
        """Predict values at given indices."""
        return self.u[row_idx] * self.v[col_idx]

    def predict_matrix(self) -> np.ndarray:
        """Predict full matrix."""
        return np.outer(self.u, self.v)

    def get_params(self) -> np.ndarray:
        """
        Get parameter vector.

        Format: [u_0, ..., u_{n_rows-1}, v_0, ..., v_{n_cols-2}]
        (last v component derived from unit norm constraint)
        """
        # Store v without last component (gauge constraint)
        return np.concatenate([self.u, self.v[:-1]])

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from vector."""
        self.u = params[:self.n_rows].copy()

        # Reconstruct v with unit norm constraint
        v_free = params[self.n_rows:]
        v_free_norm_sq = np.sum(v_free**2)

        # Last component to make ||v|| = 1
        if v_free_norm_sq >= 1.0:
            # Renormalize
            self.v = np.append(v_free, 0.0)
            self.v /= np.linalg.norm(self.v)
        else:
            v_last = np.sqrt(max(0, 1.0 - v_free_norm_sq))
            self.v = np.append(v_free, v_last)

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        # n_rows for u, n_cols - 1 for v (unit norm constraint)
        return self.n_rows + self.n_cols - 1

    def jacobian(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Analytic Jacobian for rank-1 model.

        d(u_i * v_j) / d(u_k) = v_j if i == k, else 0
        d(u_i * v_j) / d(v_k) = u_i if j == k, else 0
        Plus chain rule for unit norm constraint.
        """
        n_obs = len(row_idx)
        jac = np.zeros((n_obs, self.n_params))

        # Derivatives w.r.t. u
        for obs, (i, j) in enumerate(zip(row_idx, col_idx)):
            jac[obs, i] = self.v[j]

        # Derivatives w.r.t. v (with constraint)
        # v_k for k < n_cols - 1 are free
        # v_{n_cols-1} = sqrt(1 - sum(v_k^2))
        # d(v_last)/d(v_k) = -v_k / v_last

        v_last = self.v[-1]
        eps = 1e-10

        for obs, (i, j) in enumerate(zip(row_idx, col_idx)):
            if j < self.n_cols - 1:
                # Free parameter
                jac[obs, self.n_rows + j] = self.u[i]
            else:
                # Last component - chain rule from all free v
                if abs(v_last) > eps:
                    for k in range(self.n_cols - 1):
                        jac[obs, self.n_rows + k] += self.u[i] * (-self.v[k] / v_last)

        return jac

    @classmethod
    def from_svd(cls, M: np.ndarray, mask: Optional[np.ndarray] = None) -> "Rank1Model":
        """
        Initialize from SVD of matrix.

        Args:
            M: Matrix to approximate
            mask: Optional mask (1 = observed)

        Returns:
            Rank1Model initialized from top singular vector
        """
        n_rows, n_cols = M.shape

        # Handle missing values
        if mask is not None:
            M_filled = np.where(mask, M, 0)
        else:
            M_filled = np.nan_to_num(M, nan=0.0)

        U, s, Vt = linalg.svd(M_filled, full_matrices=False)

        # Top singular vectors
        u = U[:, 0] * np.sqrt(s[0])
        v = Vt[0, :] * np.sqrt(s[0])

        # Apply gauge: ||v|| = 1
        v_norm = np.linalg.norm(v)
        if v_norm > 0:
            u = u * v_norm
            v = v / v_norm

        model = cls(n_rows=n_rows, n_cols=n_cols)
        model.u = u
        model.v = v
        return model


@dataclass
class Rank2Model(LowRankModel):
    """
    Rank-2 matrix model: M_ij = u1_i*v1_j + u2_i*v2_j

    Gauge fixing: V = [v1, v2] has orthonormal columns

    Parameters:
    - U: (n_rows, 2) matrix of row factors
    - V: (n_cols, 2) matrix of column factors (constrained)
    """

    rank: int = field(default=2, init=False)

    U: np.ndarray = field(default=None, repr=False)
    V: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.U is None:
            self.U = np.random.randn(self.n_rows, 2)
        if self.V is None:
            self.V = np.random.randn(self.n_cols, 2)
            self.V, _ = linalg.qr(self.V, mode="economic")

    def predict(self, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
        """Predict values at given indices."""
        return np.sum(self.U[row_idx, :] * self.V[col_idx, :], axis=1)

    def predict_matrix(self) -> np.ndarray:
        """Predict full matrix."""
        return self.U @ self.V.T

    def get_params(self) -> np.ndarray:
        """
        Get parameter vector.

        Format: [U flat, V free params]
        V is orthonormal, so we parameterize it carefully.
        """
        # For simplicity, store V as-is and re-orthonormalize on set
        return np.concatenate([self.U.ravel(), self.V.ravel()])

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters and enforce orthonormality of V."""
        n_u = self.n_rows * 2
        self.U = params[:n_u].reshape(self.n_rows, 2)

        V_raw = params[n_u:].reshape(self.n_cols, 2)

        # Orthonormalize V via QR
        self.V, _ = linalg.qr(V_raw, mode="economic")

    @property
    def n_params(self) -> int:
        """Number of parameters (before constraints)."""
        return self.n_rows * 2 + self.n_cols * 2

    @property
    def n_free_params(self) -> int:
        """Number of truly free parameters after gauge fixing."""
        # U has 2*n_rows free
        # V orthonormal reduces (n_cols * 2 - 3) approximately
        # For simplicity, we use full parameterization
        return self.n_params

    @classmethod
    def from_svd(cls, M: np.ndarray, mask: Optional[np.ndarray] = None) -> "Rank2Model":
        """
        Initialize from SVD of matrix.

        Args:
            M: Matrix to approximate
            mask: Optional mask (1 = observed)

        Returns:
            Rank2Model initialized from top 2 singular vectors
        """
        n_rows, n_cols = M.shape

        if mask is not None:
            M_filled = np.where(mask, M, 0)
        else:
            M_filled = np.nan_to_num(M, nan=0.0)

        U, s, Vt = linalg.svd(M_filled, full_matrices=False)

        # Top 2 singular vectors
        U2 = U[:, :2] * np.sqrt(s[:2])
        V2 = Vt[:2, :].T * np.sqrt(s[:2])

        # Orthonormalize V2
        V2, R = linalg.qr(V2, mode="economic")
        U2 = U2 @ R  # Absorb scaling into U

        model = cls(n_rows=n_rows, n_cols=n_cols)
        model.U = U2
        model.V = V2
        return model


def compute_ndof(model: LowRankModel, n_obs: int) -> int:
    """
    Compute number of degrees of freedom.

    ndof = n_observations - n_free_parameters
    """
    return n_obs - model.n_params


def compute_frobenius_distance(
    M_true: np.ndarray,
    M_approx: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute (weighted) Frobenius distance between matrices.

    Args:
        M_true: True matrix
        M_approx: Approximation
        weights: Optional weights (same shape as M)

    Returns:
        Weighted Frobenius norm of difference
    """
    diff = M_true - M_approx

    if weights is not None:
        return np.sqrt(np.sum(weights * diff**2))
    else:
        return np.linalg.norm(diff, "fro")
