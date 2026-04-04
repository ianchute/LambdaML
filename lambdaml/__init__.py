"""
LambdaML — gradient-free machine learning for any numpy-compatible function.

Quick start
-----------
>>> from lambdaml import LambdaClassifierModel, LambdaRegressorModel
>>> from lambdaml import Optimizer, DiffMethod, LRSchedule
"""

from .lambda_model import (
    LambdaClassifierModel,
    LambdaRegressorModel,
    Optimizer,
)
from .lambda_utils import (
    DiffMethod,
    NumericalDiff,
    GradientComputer,
    Regularization,
    LossFunctions,
    LRSchedule,
)

__all__ = [
    "LambdaClassifierModel",
    "LambdaRegressorModel",
    "Optimizer",
    "DiffMethod",
    "NumericalDiff",
    "GradientComputer",
    "Regularization",
    "LossFunctions",
    "LRSchedule",
]

__version__ = "1.1.0"
__author__ = "Ian Chu Te"
