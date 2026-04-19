"""
LambdaML — gradient-free machine learning for any numpy-compatible function.

Quick start
-----------
>>> from lambdaml import LambdaClassifierModel, LambdaRegressorModel
>>> from lambdaml import Optimizer, DiffMethod, LRSchedule

ONNX export
-----------
>>> proto = model.to_onnx('model.onnx', input_shape=(2,))   # vectorized models
>>> model.save_params('weights.npz')                         # always works
>>> from lambdaml import predict_onnx, from_onnx
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
from .lambda_onnx import (
    to_onnx,
    from_onnx,
    save_params,
    load_params,
    predict_onnx,
    OnnxTraceError,
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
    # ONNX
    "to_onnx",
    "from_onnx",
    "save_params",
    "load_params",
    "predict_onnx",
    "OnnxTraceError",
]

__version__ = "1.2.0"
__author__ = "Ian Chu Te"
