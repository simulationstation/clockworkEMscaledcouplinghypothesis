"""
Low-rank matrix fitting algorithms.

Implements two fitting backends:
1. Weighted Alternating Least Squares (ALS) for diagonal weights
2. Nonlinear Least Squares for full covariance

Also includes fit health checks and multi-start stability analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

import numpy as np
from scipy import linalg
from scipy.optimize import least_squares, OptimizeResult

from rank1.models.lowrank import LowRankModel, Rank1Model, Rank2Model
from rank1.logging import get_logger

logger = get_logger()


class FitMethod(Enum):
    """Available fitting methods."""
    ALS = "als"
    NLLS = "nlls"  # Nonlinear least squares
    AUTO = "auto"


@dataclass
class FitHealthCheck:
    """
    Results of fit health diagnostics.

    Tracks multiple health criteria to identify problematic fits:
    - Convergence: Did optimizer terminate normally?
    - Gradient norm: Is gradient close to zero at solution?
    - Chi2/ndof: Is reduced chi2 in reasonable range?
    - Condition number: Is Jacobian well-conditioned?
    - Stability: Do multiple starts give same solution?
    - Hessian: Is Hessian positive definite at solution?
    """

    converged: bool = True
    gradient_norm: float = 0.0
    condition_number: float = 0.0
    chi2_ndof: float = 0.0

    # Problem flags
    is_underconstrained: bool = False
    is_catastrophically_bad: bool = False
    is_stable: bool = True
    is_hessian_ok: bool = True
    is_gradient_ok: bool = True

    # Thresholds used (for logging)
    thresholds: dict = field(default_factory=dict)

    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all critical health criteria pass."""
        return (
            self.converged and
            not self.is_underconstrained and
            not self.is_catastrophically_bad and
            self.is_stable and
            self.is_hessian_ok
        )

    @property
    def passed_strict(self) -> bool:
        """Check if ALL health criteria pass (including warnings)."""
        return self.passed and self.is_gradient_ok and len(self.warnings) == 0

    @property
    def severity(self) -> str:
        """Get severity level: 'ok', 'warning', 'error'."""
        if self.is_catastrophically_bad or not self.converged:
            return "error"
        if not self.passed:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "passed_strict": self.passed_strict,
            "severity": self.severity,
            "converged": self.converged,
            "gradient_norm": self.gradient_norm,
            "condition_number": self.condition_number,
            "chi2_ndof": self.chi2_ndof,
            "is_underconstrained": self.is_underconstrained,
            "is_catastrophically_bad": self.is_catastrophically_bad,
            "is_stable": self.is_stable,
            "is_hessian_ok": self.is_hessian_ok,
            "is_gradient_ok": self.is_gradient_ok,
            "thresholds": self.thresholds,
            "messages": self.messages,
            "warnings": self.warnings,
        }


@dataclass
class FitResult:
    """Result of low-rank model fitting."""

    model: LowRankModel
    chi2: float
    ndof: int
    n_obs: int

    residuals: np.ndarray
    jacobian: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None

    n_iterations: int = 0
    success: bool = True
    message: str = ""

    health: Optional[FitHealthCheck] = None

    @property
    def chi2_ndof(self) -> float:
        """Reduced chi-squared."""
        if self.ndof > 0:
            return self.chi2 / self.ndof
        return np.nan

    @property
    def predicted_matrix(self) -> np.ndarray:
        """Get predicted matrix from model."""
        return self.model.predict_matrix()


class LowRankFitter:
    """
    Fit low-rank models to matrix data.

    Supports both ALS (fast for diagonal weights) and nonlinear
    least squares (for full covariance).
    """

    def __init__(
        self,
        method: FitMethod = FitMethod.AUTO,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        regularization: float = 1e-8,
        n_starts: int = 10,
        check_health: bool = True,
    ):
        """
        Initialize fitter.

        Args:
            method: Fitting method (ALS, NLLS, or AUTO)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            regularization: Ridge regularization
            n_starts: Number of random starts for stability
            check_health: Run fit health diagnostics
        """
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.n_starts = n_starts
        self.check_health = check_health

    def fit(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        rank: int = 1,
        covariance: Optional[np.ndarray] = None,
        initial_model: Optional[LowRankModel] = None,
    ) -> FitResult:
        """
        Fit a low-rank model to sparse matrix observations.

        Args:
            row_idx: Row indices of observations
            col_idx: Column indices of observations
            values: Observed values
            errors: Observation uncertainties (1D for diagonal, or full covariance)
            n_rows: Number of matrix rows
            n_cols: Number of matrix columns
            rank: Target rank (1 or 2)
            covariance: Full covariance matrix (optional)
            initial_model: Initial model guess

        Returns:
            FitResult with fitted model and diagnostics
        """
        n_obs = len(values)

        # Select fitting method
        method = self.method
        if method == FitMethod.AUTO:
            if covariance is not None:
                method = FitMethod.NLLS
            else:
                method = FitMethod.ALS

        # Initialize model
        if initial_model is not None:
            model = initial_model
        else:
            # Ensure indices are integer type
            row_idx = np.asarray(row_idx, dtype=np.intp)
            col_idx = np.asarray(col_idx, dtype=np.intp)
            # Build matrix for SVD initialization
            M = np.full((n_rows, n_cols), np.nan)
            M[row_idx, col_idx] = values

            if rank == 1:
                model = Rank1Model.from_svd(M)
            else:
                model = Rank2Model.from_svd(M)

        # Fit
        if method == FitMethod.ALS:
            result = self._fit_als(
                model, row_idx, col_idx, values, errors
            )
        else:
            result = self._fit_nlls(
                model, row_idx, col_idx, values, errors, covariance
            )

        # Health check
        if self.check_health:
            result.health = self._check_fit_health(result, errors)

        return result

    def fit_rank1(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        **kwargs,
    ) -> FitResult:
        """Convenience method for rank-1 fitting."""
        return self.fit(
            row_idx, col_idx, values, errors,
            n_rows, n_cols, rank=1, **kwargs
        )

    def fit_rank2(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        **kwargs,
    ) -> FitResult:
        """Convenience method for rank-2 fitting."""
        return self.fit(
            row_idx, col_idx, values, errors,
            n_rows, n_cols, rank=2, **kwargs
        )

    def _fit_als(
        self,
        model: LowRankModel,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
    ) -> FitResult:
        """
        Weighted Alternating Least Squares.

        For rank-1: alternates between solving for u and v.
        For rank-2: alternates between U and V.
        """
        n_obs = len(values)
        weights = 1.0 / (errors**2 + self.regularization)

        prev_cost = np.inf
        n_iter = 0
        converged = False

        for n_iter in range(self.max_iterations):
            if isinstance(model, Rank1Model):
                # Update u given v
                self._als_update_u_rank1(model, row_idx, col_idx, values, weights)

                # Update v given u
                self._als_update_v_rank1(model, row_idx, col_idx, values, weights)

            elif isinstance(model, Rank2Model):
                # Update U given V
                self._als_update_u_rank2(model, row_idx, col_idx, values, weights)

                # Update V given U (with orthonormalization)
                self._als_update_v_rank2(model, row_idx, col_idx, values, weights)

            # Check convergence
            residuals = values - model.predict(row_idx, col_idx)
            cost = np.sum(weights * residuals**2)

            if abs(prev_cost - cost) < self.tolerance * max(1, abs(cost)):
                converged = True
                break

            prev_cost = cost

        # Final residuals and chi2
        residuals = values - model.predict(row_idx, col_idx)
        chi2 = float(np.sum((residuals / errors)**2))
        ndof = n_obs - model.n_params

        return FitResult(
            model=model,
            chi2=chi2,
            ndof=ndof,
            n_obs=n_obs,
            residuals=residuals,
            n_iterations=n_iter + 1,
            success=converged,
            message="ALS converged" if converged else "ALS max iterations reached",
        )

    def _als_update_u_rank1(
        self,
        model: Rank1Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Update u in rank-1 ALS."""
        # For each row i: u_i = sum_j(w_ij * v_j * y_ij) / sum_j(w_ij * v_j^2)
        for i in range(model.n_rows):
            mask = row_idx == i
            if not np.any(mask):
                continue

            j_vals = col_idx[mask]
            y_vals = values[mask]
            w_vals = weights[mask]
            v_vals = model.v[j_vals]

            numer = np.sum(w_vals * v_vals * y_vals)
            denom = np.sum(w_vals * v_vals**2) + self.regularization

            model.u[i] = numer / denom

    def _als_update_v_rank1(
        self,
        model: Rank1Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Update v in rank-1 ALS, with unit norm constraint."""
        # First compute unconstrained v
        v_new = np.zeros(model.n_cols)

        for j in range(model.n_cols):
            mask = col_idx == j
            if not np.any(mask):
                v_new[j] = 0
                continue

            i_vals = row_idx[mask]
            y_vals = values[mask]
            w_vals = weights[mask]
            u_vals = model.u[i_vals]

            numer = np.sum(w_vals * u_vals * y_vals)
            denom = np.sum(w_vals * u_vals**2) + self.regularization

            v_new[j] = numer / denom

        # Apply unit norm constraint, absorb scale into u
        v_norm = np.linalg.norm(v_new)
        if v_norm > 0:
            model.u *= v_norm
            model.v = v_new / v_norm
        else:
            model.v = v_new

    def _als_update_u_rank2(
        self,
        model: Rank2Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Update U in rank-2 ALS."""
        for i in range(model.n_rows):
            mask = row_idx == i
            if not np.any(mask):
                continue

            j_vals = col_idx[mask]
            y_vals = values[mask]
            w_vals = weights[mask]
            V_sub = model.V[j_vals, :]  # (n_j, 2)

            # Weighted normal equations
            W = np.diag(w_vals)
            A = V_sub.T @ W @ V_sub + self.regularization * np.eye(2)
            b = V_sub.T @ (w_vals * y_vals)

            model.U[i, :] = linalg.solve(A, b)

    def _als_update_v_rank2(
        self,
        model: Rank2Model,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Update V in rank-2 ALS with orthonormalization."""
        V_new = np.zeros((model.n_cols, 2))

        for j in range(model.n_cols):
            mask = col_idx == j
            if not np.any(mask):
                continue

            i_vals = row_idx[mask]
            y_vals = values[mask]
            w_vals = weights[mask]
            U_sub = model.U[i_vals, :]

            W = np.diag(w_vals)
            A = U_sub.T @ W @ U_sub + self.regularization * np.eye(2)
            b = U_sub.T @ (w_vals * y_vals)

            V_new[j, :] = linalg.solve(A, b)

        # Orthonormalize and absorb transform into U
        V_new, R = linalg.qr(V_new, mode="economic")
        model.V = V_new
        model.U = model.U @ R

    def _fit_nlls(
        self,
        model: LowRankModel,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        covariance: Optional[np.ndarray] = None,
    ) -> FitResult:
        """
        Nonlinear least squares fitting.

        Uses scipy.optimize.least_squares with optional whitening
        for full covariance.
        """
        n_obs = len(values)

        # Compute whitening transform if full covariance
        if covariance is not None:
            L = linalg.cholesky(
                covariance + self.regularization * np.eye(n_obs),
                lower=True
            )
            L_inv = linalg.solve_triangular(L, np.eye(n_obs), lower=True)
        else:
            L_inv = np.diag(1.0 / errors)

        def residual_func(params):
            model.set_params(params)
            pred = model.predict(row_idx, col_idx)
            resid = values - pred
            # Whiten residuals
            return L_inv @ resid

        def jacobian_func(params):
            model.set_params(params)
            jac = model.jacobian(row_idx, col_idx)
            # Transform Jacobian
            return -L_inv @ jac

        # Run optimization
        x0 = model.get_params()

        result = least_squares(
            residual_func,
            x0,
            jac=jacobian_func,
            method="lm",
            max_nfev=self.max_iterations * len(x0),
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )

        model.set_params(result.x)

        # Compute chi2
        residuals = values - model.predict(row_idx, col_idx)
        if covariance is not None:
            whitened = L_inv @ residuals
            chi2 = float(np.sum(whitened**2))
        else:
            chi2 = float(np.sum((residuals / errors)**2))

        ndof = n_obs - model.n_params

        return FitResult(
            model=model,
            chi2=chi2,
            ndof=ndof,
            n_obs=n_obs,
            residuals=residuals,
            jacobian=result.jac if hasattr(result, "jac") else None,
            n_iterations=result.nfev,
            success=result.success,
            message=result.message,
        )

    def _check_fit_health(
        self,
        result: FitResult,
        errors: np.ndarray,
        chi2_ndof_lower: float = 0.1,
        chi2_ndof_upper: float = 10.0,
        chi2_ndof_warning_lower: float = 0.3,
        chi2_ndof_warning_upper: float = 3.0,
        condition_max: float = 1e10,
        condition_warning: float = 1e6,
        gradient_max: float = 1e-3,
        gradient_warning: float = 1e-6,
    ) -> FitHealthCheck:
        """
        Run comprehensive fit health diagnostics.

        Checks multiple criteria and provides both error-level and
        warning-level flags.

        Args:
            result: Fit result to check
            errors: Observation errors
            chi2_ndof_lower: Chi2/ndof below this is ERROR (suspiciously good)
            chi2_ndof_upper: Chi2/ndof above this is ERROR (catastrophic)
            chi2_ndof_warning_lower: Chi2/ndof below this is WARNING
            chi2_ndof_warning_upper: Chi2/ndof above this is WARNING
            condition_max: Condition number above this is ERROR
            condition_warning: Condition number above this is WARNING
            gradient_max: Gradient norm above this is ERROR
            gradient_warning: Gradient norm above this is WARNING

        Returns:
            FitHealthCheck with all diagnostics
        """
        health = FitHealthCheck()

        # Store thresholds for transparency
        health.thresholds = {
            "chi2_ndof_lower": chi2_ndof_lower,
            "chi2_ndof_upper": chi2_ndof_upper,
            "chi2_ndof_warning_lower": chi2_ndof_warning_lower,
            "chi2_ndof_warning_upper": chi2_ndof_warning_upper,
            "condition_max": condition_max,
            "condition_warning": condition_warning,
            "gradient_max": gradient_max,
            "gradient_warning": gradient_warning,
        }

        # 1. Convergence check
        health.converged = result.success
        if not result.success:
            health.messages.append(f"Optimizer did not converge: {result.message}")
            logger.warning(f"FIT HEALTH: Convergence failure - {result.message}")

        # 2. Chi2/ndof checks
        health.chi2_ndof = result.chi2_ndof

        if result.ndof > 0:
            # Error level
            if health.chi2_ndof < chi2_ndof_lower:
                health.is_underconstrained = True
                msg = (
                    f"chi2/ndof = {health.chi2_ndof:.3f} < {chi2_ndof_lower} "
                    "(suspiciously good - possible overfitting or error overestimation)"
                )
                health.messages.append(msg)
                logger.warning(f"FIT HEALTH: {msg}")

            if health.chi2_ndof > chi2_ndof_upper:
                health.is_catastrophically_bad = True
                msg = (
                    f"chi2/ndof = {health.chi2_ndof:.3f} > {chi2_ndof_upper} "
                    "(very poor fit - model may be inadequate or errors underestimated)"
                )
                health.messages.append(msg)
                logger.warning(f"FIT HEALTH: {msg}")

            # Warning level
            if chi2_ndof_lower < health.chi2_ndof < chi2_ndof_warning_lower:
                health.warnings.append(
                    f"chi2/ndof = {health.chi2_ndof:.3f} is low (expected ~1.0)"
                )

            if chi2_ndof_warning_upper < health.chi2_ndof < chi2_ndof_upper:
                health.warnings.append(
                    f"chi2/ndof = {health.chi2_ndof:.3f} is elevated (expected ~1.0)"
                )

        # 3. Condition number of Jacobian
        if result.jacobian is not None:
            try:
                s = linalg.svdvals(result.jacobian)
                if len(s) > 0 and s[-1] > 0:
                    health.condition_number = float(s[0] / s[-1])

                    if health.condition_number > condition_max:
                        health.is_underconstrained = True
                        msg = (
                            f"Condition number = {health.condition_number:.2e} "
                            "(numerically unstable - problem may be ill-posed)"
                        )
                        health.messages.append(msg)
                        logger.warning(f"FIT HEALTH: {msg}")
                    elif health.condition_number > condition_warning:
                        health.warnings.append(
                            f"Condition number = {health.condition_number:.2e} is elevated"
                        )

                    # Check for near-zero singular values
                    if s[-1] < 1e-12:
                        health.is_underconstrained = True
                        health.messages.append("Near-zero singular value in Jacobian")
            except Exception as e:
                health.warnings.append(f"Could not compute condition number: {e}")

        # 4. Gradient norm at solution
        if result.jacobian is not None and result.residuals is not None:
            try:
                # Gradient of chi2 = 2 * J^T @ residuals / sigma^2
                grad = 2.0 * result.jacobian.T @ (result.residuals / errors**2)
                health.gradient_norm = float(np.linalg.norm(grad))

                if health.gradient_norm > gradient_max:
                    health.is_gradient_ok = False
                    msg = (
                        f"Gradient norm = {health.gradient_norm:.2e} > {gradient_max} "
                        "(solution may not be at local minimum)"
                    )
                    health.messages.append(msg)
                    logger.warning(f"FIT HEALTH: {msg}")
                elif health.gradient_norm > gradient_warning:
                    health.warnings.append(
                        f"Gradient norm = {health.gradient_norm:.2e} is elevated"
                    )
            except Exception as e:
                health.warnings.append(f"Could not compute gradient norm: {e}")

        # 5. Check Hessian positive definiteness (approximate)
        if result.jacobian is not None:
            try:
                # Approximate Hessian via Gauss-Newton: H ≈ 2 * J^T @ J
                JTJ = result.jacobian.T @ result.jacobian
                eigvals = linalg.eigvalsh(JTJ)

                if np.any(eigvals < 0):
                    health.is_hessian_ok = False
                    health.messages.append(
                        "Hessian has negative eigenvalues (not at local minimum)"
                    )
                elif np.any(eigvals < 1e-12):
                    health.warnings.append(
                        "Hessian has near-zero eigenvalues (flat direction)"
                    )
            except Exception as e:
                health.warnings.append(f"Could not check Hessian: {e}")

        # 6. Check for extreme parameter values
        params = result.model.get_params()
        if np.any(np.abs(params) > 1e6):
            health.warnings.append(
                f"Some parameters are very large (max = {np.max(np.abs(params)):.2e})"
            )
        if np.any(~np.isfinite(params)):
            health.is_catastrophically_bad = True
            health.messages.append("Non-finite parameter values")

        # Log summary
        if health.passed:
            if health.warnings:
                logger.info(
                    f"FIT HEALTH: Passed with {len(health.warnings)} warning(s)"
                )
        else:
            logger.warning(
                f"FIT HEALTH: FAILED - {len(health.messages)} issue(s)"
            )

        return health

    def fit_multistart(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: np.ndarray,
        errors: np.ndarray,
        n_rows: int,
        n_cols: int,
        rank: int = 1,
        n_starts: Optional[int] = None,
        seed: int = 42,
    ) -> tuple[FitResult, list[FitResult]]:
        """
        Fit with multiple random initializations.

        Returns the best fit and list of all fits for stability analysis.

        Args:
            row_idx, col_idx, values, errors: Observation data
            n_rows, n_cols: Matrix dimensions
            rank: Target rank
            n_starts: Number of random starts (uses self.n_starts if None)
            seed: Random seed

        Returns:
            (best_result, all_results) tuple
        """
        if n_starts is None:
            n_starts = self.n_starts

        rng = np.random.default_rng(seed)
        all_results = []

        for i in range(n_starts):
            # Create random initialization
            if rank == 1:
                model = Rank1Model(n_rows=n_rows, n_cols=n_cols)
                model.u = rng.standard_normal(n_rows)
                model.v = rng.standard_normal(n_cols)
                model.v /= np.linalg.norm(model.v)
            else:
                model = Rank2Model(n_rows=n_rows, n_cols=n_cols)
                model.U = rng.standard_normal((n_rows, 2))
                model.V = rng.standard_normal((n_cols, 2))
                model.V, _ = linalg.qr(model.V, mode="economic")

            try:
                result = self.fit(
                    row_idx, col_idx, values, errors,
                    n_rows, n_cols, rank=rank,
                    initial_model=model,
                )
                all_results.append(result)
            except Exception as e:
                logger.debug(f"Multistart iteration {i} failed: {e}")

        if not all_results:
            raise RuntimeError("All multistart iterations failed")

        # Sort by chi2
        all_results.sort(key=lambda r: r.chi2)
        best = all_results[0]

        # Check stability
        if len(all_results) >= 3:
            best.health = self._check_multistart_stability(all_results)

        return best, all_results

    def _check_multistart_stability(
        self,
        results: list[FitResult],
        tolerance: float = 0.05,
    ) -> FitHealthCheck:
        """Check if multiple starts give consistent results."""
        health = FitHealthCheck()

        # Compare predicted matrices from top 3 fits
        matrices = [r.predicted_matrix for r in results[:3]]

        diffs = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                rel_diff = np.linalg.norm(matrices[i] - matrices[j]) / (
                    np.linalg.norm(matrices[i]) + 1e-10
                )
                diffs.append(rel_diff)

        max_diff = max(diffs) if diffs else 0

        if max_diff > tolerance:
            health.is_stable = False
            health.messages.append(
                f"Multistart solutions differ by {max_diff:.1%} "
                "(unstable optimization)"
            )

        return health
