"""Dataset classes for rank-1 factorization analysis."""

from rank1.datasets.base import MatrixDataset, MatrixObservation
from rank1.datasets.higgs_atlas_mu import HiggsATLASDataset
from rank1.datasets.elastic_totem import ElasticTOTEMDataset
from rank1.datasets.diffractive_dis import DiffractiveDISDataset

__all__ = [
    "MatrixDataset",
    "MatrixObservation",
    "HiggsATLASDataset",
    "ElasticTOTEMDataset",
    "DiffractiveDISDataset",
]
