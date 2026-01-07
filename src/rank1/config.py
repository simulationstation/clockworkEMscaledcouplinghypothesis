"""
Configuration management using Pydantic models.

Provides centralized configuration for:
- Data directories and caching
- Analysis parameters (bootstrap iterations, tolerances)
- Fit health gates and convergence criteria
- Parallel execution settings
"""

from pathlib import Path
from typing import Optional
import json
import yaml

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data directories and caching."""

    base_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    cache_dir: Path = Field(default=Path("data/cache"))
    processed_dir: Path = Field(default=Path("data/processed"))

    cache_ttl_hours: int = Field(default=24 * 7, description="HTTP cache TTL in hours")
    verify_checksums: bool = Field(default=True)

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [self.raw_dir, self.cache_dir, self.processed_dir]:
            d.mkdir(parents=True, exist_ok=True)


class FitHealthConfig(BaseModel):
    """Configuration for fit health gates and convergence criteria."""

    # Convergence criteria
    gradient_tol: float = Field(default=1e-6, description="Gradient norm tolerance")
    cost_tol: float = Field(default=1e-10, description="Relative cost change tolerance")
    max_iterations: int = Field(default=1000, description="Maximum optimizer iterations")

    # Underconstrained detection
    chi2_ndof_lower: float = Field(default=0.1, description="Lower bound for chi2/ndof (too good)")
    condition_number_max: float = Field(default=1e10, description="Max condition number")

    # Catastrophically bad detection
    chi2_ndof_upper: float = Field(default=10.0, description="Upper bound for chi2/ndof")

    # Multi-start stability
    n_starts: int = Field(default=10, description="Number of random initializations")
    stability_tol: float = Field(default=0.05, description="Relative tolerance for stability check")


class BootstrapConfig(BaseModel):
    """Configuration for parametric bootstrap."""

    n_bootstrap: int = Field(default=1000, description="Number of bootstrap iterations")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    use_parallel: bool = Field(default=True, description="Use parallel execution")

    @field_validator("n_bootstrap")
    @classmethod
    def validate_n_bootstrap(cls, v: int) -> int:
        if v < 50:
            raise ValueError("n_bootstrap should be at least 50 for meaningful p-values")
        return v


class ParallelConfig(BaseModel):
    """Configuration for parallel execution."""

    n_jobs: int = Field(default=-1, description="-1 means use all but one CPU")
    backend: str = Field(default="loky", description="Joblib backend")
    batch_size: str = Field(default="auto", description="Batch size for joblib")

    def get_n_workers(self) -> int:
        """Get actual number of workers to use."""
        import os
        n_cpus = os.cpu_count() or 1
        if self.n_jobs == -1:
            return max(1, n_cpus - 1)
        elif self.n_jobs < 0:
            return max(1, n_cpus + 1 + self.n_jobs)
        else:
            return min(self.n_jobs, n_cpus)


class AnalysisConfig(BaseModel):
    """Configuration for a specific analysis run."""

    name: str
    dataset: str

    # Fitting options
    use_full_covariance: bool = Field(default=False)
    regularization: float = Field(default=1e-8, description="Ridge regularization")

    # Bootstrap
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    # Fit health
    fit_health: FitHealthConfig = Field(default_factory=FitHealthConfig)

    # Output options
    save_figures: bool = Field(default=True)
    figure_format: list[str] = Field(default=["png", "pdf"])
    save_intermediate: bool = Field(default=False)


class Config(BaseModel):
    """Top-level configuration for the rank-1 analysis pipeline."""

    # Data settings
    data: DataConfig = Field(default_factory=DataConfig)

    # Parallel execution
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)

    # Output directory
    output_dir: Path = Field(default=Path("outputs"))

    # Global seed
    seed: int = Field(default=42)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data.ensure_dirs()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_analysis_config(
        self,
        name: str,
        dataset: str,
        **overrides: dict
    ) -> AnalysisConfig:
        """Create an analysis configuration with defaults from this config."""
        bootstrap_cfg = BootstrapConfig(
            seed=self.seed,
            use_parallel=True,
            **overrides.pop("bootstrap", {})
        )
        fit_health_cfg = FitHealthConfig(**overrides.pop("fit_health", {}))

        return AnalysisConfig(
            name=name,
            dataset=dataset,
            bootstrap=bootstrap_cfg,
            fit_health=fit_health_cfg,
            **overrides
        )

    def save(self, path: Path) -> None:
        """Save configuration to file (YAML or JSON)."""
        data = self.model_dump(mode="json")

        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj

        data = convert_paths(data)

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in (".yml", ".yaml"):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from file."""
        with open(path) as f:
            if path.suffix in (".yml", ".yaml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)


# Default configuration singleton
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set the default configuration instance."""
    global _default_config
    _default_config = config
