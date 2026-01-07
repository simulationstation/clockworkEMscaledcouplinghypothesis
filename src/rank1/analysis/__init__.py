"""Analysis modules for rank-1 factorization tests."""

from rank1.analysis.base import BaseRankAnalysis, AnalysisResult
from rank1.analysis.higgs_rank_test import HiggsRankAnalysis
from rank1.analysis.elastic_rank_test import ElasticRankAnalysis
from rank1.analysis.diffractive_rank_test import DiffractiveRankAnalysis

# NP (New Physics Sensitive) analysis
from rank1.analysis.np_analysis import NPAnalyzer, NPResult, NPVerdict
from rank1.analysis.residual_mode import (
    ResidualMode,
    ResidualMap,
    ResidualModeExtractor,
    LocalizationMetrics,
    StabilityMetrics,
)
from rank1.analysis.sweeps import (
    SweepPreset,
    SweepResult,
    SweepRunner,
    GlobalSignificance,
    get_presets_for_dataset,
    get_fast_presets,
)
from rank1.analysis.replication import (
    ReplicationMetrics,
    ReplicationReport,
    ModeComparator,
)

__all__ = [
    # Base analysis
    "BaseRankAnalysis",
    "AnalysisResult",
    "HiggsRankAnalysis",
    "ElasticRankAnalysis",
    "DiffractiveRankAnalysis",
    # NP analysis
    "NPAnalyzer",
    "NPResult",
    "NPVerdict",
    "ResidualMode",
    "ResidualMap",
    "ResidualModeExtractor",
    "LocalizationMetrics",
    "StabilityMetrics",
    "SweepPreset",
    "SweepResult",
    "SweepRunner",
    "GlobalSignificance",
    "get_presets_for_dataset",
    "get_fast_presets",
    "ReplicationMetrics",
    "ReplicationReport",
    "ModeComparator",
]
