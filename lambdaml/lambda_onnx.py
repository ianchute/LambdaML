"""
lambda_onnx.py
--------------
ONNX export and import for LambdaML models.

Overview
--------
LambdaML lets you write ``f(x, p)`` as *any* numpy-compatible function.
Exporting to ONNX requires tracing the computation graph — which is only
possible when the function uses a traceable subset of numpy operations.

Two export strategies are provided:

1. **Auto-trace** (``to_onnx()``)
   Wraps the learned parameters as ONNX initializers and traces a single
   vectorized forward pass using ``onnxscript``'s numpy-compatible IR.
   Works for functions built from standard numpy ops (matmul, add, exp,
   tanh, sin, cos, …).  Requires ``vectorized=True`` on the model.

2. **Parameters-only / npz** (``save_params()`` / ``load_params()``)
   Always works — no tracing needed.  Saves the parameter dict (and
   optional metadata) as a compressed ``.npz`` file.  On load you must
   reconstruct the model and call ``load_params()`` to restore weights.

Quick start
-----------
**Auto-trace export** (vectorized models only):

>>> def logistic_v(X, p):            # vectorized: X is (n, d)
...     return 1 / (1 + np.exp(-(X @ p['w'] + p['b'])))
...
>>> model = LambdaClassifierModel(f=logistic_v, p={'w': np.zeros(2), 'b': 0.0},
...                               vectorized=True)
>>> model.fit(X_train, Y_train, n_iter=200, lr=0.01)
>>> model.to_onnx('logistic.onnx', input_shape=(2,))
>>> preds = model.predict_onnx('logistic.onnx', X_test)

**Params-only save/load** (always works):

>>> model.save_params('my_model.npz')
>>> # Later, in a new session:
>>> model2 = LambdaClassifierModel(f=logistic_v, p={'w': np.zeros(2), 'b': 0.0},
...                                vectorized=True)
>>> model2.load_params('my_model.npz')
>>> preds = model2.predict(X_test)

ONNX Runtime inference
-----------------------
>>> import onnxruntime as rt
>>> sess = rt.InferenceSession('logistic.onnx')
>>> input_name = sess.get_inputs()[0].name
>>> probs = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

Notes
-----
- ``to_onnx()`` requires ``pip install lambdaml[onnx]``
  (installs ``onnx`` and ``onnxruntime``).
- ``onnxscript`` is an optional accelerator for tracing; if absent, a
  lightweight built-in tracer is used for common ops.
- Parameters are embedded as ONNX initializers (float32 weights baked in).
- For non-traceable functions, use ``save_params()`` / ``load_params()``
  and keep your model function in source.
"""

from __future__ import annotations

import io
import warnings
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lambda_model import _LambdaBaseModel

__all__ = [
    "to_onnx",
    "from_onnx",
    "save_params",
    "load_params",
    "predict_onnx",
    "OnnxTraceError",
]


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class OnnxTraceError(RuntimeError):
    """
    Raised when ONNX tracing fails for a given model function.

    Common causes
    -------------
    - The model was created with ``vectorized=False``  (tracing requires the
      vectorized interface ``f(X, p)`` where X is a 2-D matrix).
    - The function uses Python control flow that depends on tensor values
      (if/while branching on array contents).
    - The function calls a numpy op that has no ONNX equivalent
      (e.g. ``np.linalg.solve``, sparse operations, custom C extensions).

    Workaround
    ----------
    Use ``save_params()`` / ``load_params()`` to persist weights and
    reconstruct the model from source at load time.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ──────────────────────────────────────────────────────────────────────────────

def _require_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export/import.\n"
            "Install it with:  pip install lambdaml[onnx]\n"
            "  or directly:    pip install onnx onnxruntime"
        ) from None


def _require_onnxruntime():
    try:
        import onnxruntime as rt
        return rt
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference.\n"
            "Install it with:  pip install lambdaml[onnx]\n"
            "  or directly:    pip install onnxruntime"
        ) from None


# ──────────────────────────────────────────────────────────────────────────────
# Numpy-to-ONNX tracer
# ──────────────────────────────────────────────────────────────────────────────

class _TracerArray:
    """
    A proxy numpy array that records ONNX ops instead of computing them.

    Each arithmetic operation appends an ONNX node to ``_nodes`` and
    returns a new ``_TracerArray`` representing the output tensor.
    """

    _node_counter = 0

    def __init__(self, name: str, shape, dtype=np.float32):
        self.name  = name
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def _fresh(cls, shape, dtype=np.float32, prefix="t") -> "_TracerArray":
        cls._node_counter += 1
        return cls(f"{prefix}_{cls._node_counter}", shape, dtype)

    # ── unary ops ─────────────────────────────────────────────────────────────

    def _unary(self, op_type: str) -> "_TracerArray":
        out = _TracerArray._fresh(self.shape, self.dtype)
        _TraceContext.current().add_node(op_type, [self.name], [out.name])
        return out

    def __neg__(self):
        return self._unary("Neg")

    # ── binary ops ────────────────────────────────────────────────────────────

    def _binary(self, other, op_type: str) -> "_TracerArray":
        ctx = _TraceContext.current()
        rhs_name = ctx.constant(other) if not isinstance(other, _TracerArray) else other.name
        rhs_shape = other.shape if isinstance(other, _TracerArray) else ()
        # Compute output shape safely — replace None dims with 1 for broadcast calc,
        # then restore None for any dynamic dim in the result.
        if rhs_shape:
            def _safe_broadcast(s1, s2):
                # Replace None with 1 for numpy broadcast, track which dims were None
                s1f = tuple(1 if d is None else d for d in s1)
                s2f = tuple(1 if d is None else d for d in s2)
                try:
                    out = list(np.broadcast_shapes(s1f, s2f))
                except (TypeError, ValueError):
                    out = list(s1f)
                # Any position that was None in either input → None in output
                pad1 = (len(out) - len(s1)) * [1] + list(s1)
                pad2 = (len(out) - len(s2)) * [1] + list(s2)
                for i, (a, b) in enumerate(zip(pad1, pad2)):
                    if a is None or b is None:
                        out[i] = None
                return tuple(out)
            out_shape = _safe_broadcast(self.shape, rhs_shape)
        else:
            out_shape = self.shape
        out = _TracerArray._fresh(out_shape, self.dtype)
        ctx.add_node(op_type, [self.name, rhs_name], [out.name])
        return out

    def __add__(self, other):  return self._binary(other, "Add")
    def __radd__(self, other): return self._binary(other, "Add")
    def __sub__(self, other):  return self._binary(other, "Sub")
    def __rsub__(self, other):
        ctx = _TraceContext.current()
        lhs_name = ctx.constant(other)
        out = _TracerArray._fresh(self.shape, self.dtype)
        ctx.add_node("Sub", [lhs_name, self.name], [out.name])
        return out
    def __mul__(self, other):  return self._binary(other, "Mul")
    def __rmul__(self, other): return self._binary(other, "Mul")
    def __truediv__(self, other):  return self._binary(other, "Div")
    def __rtruediv__(self, other):
        ctx = _TraceContext.current()
        lhs_name = ctx.constant(other)
        out = _TracerArray._fresh(self.shape, self.dtype)
        ctx.add_node("Div", [lhs_name, self.name], [out.name])
        return out
    def __pow__(self, other):  return self._binary(other, "Pow")

    # ── matmul ────────────────────────────────────────────────────────────────

    def __matmul__(self, other):
        ctx = _TraceContext.current()
        rhs_name = ctx.constant(other) if not isinstance(other, _TracerArray) else other.name
        # output shape: drop last dim of lhs, replace with last dim of rhs
        if isinstance(other, _TracerArray):
            out_shape = self.shape[:-1] + other.shape[-1:]
        else:
            arr = np.asarray(other)
            out_shape = self.shape[:-1] + arr.shape[-1:]
        out = _TracerArray._fresh(out_shape, self.dtype)
        ctx.add_node("MatMul", [self.name, rhs_name], [out.name])
        return out

    def __rmatmul__(self, other):
        ctx = _TraceContext.current()
        lhs_name = ctx.constant(other) if not isinstance(other, _TracerArray) else other.name
        arr = np.asarray(other)
        out_shape = arr.shape[:-1] + self.shape[-1:]
        out = _TracerArray._fresh(out_shape, self.dtype)
        ctx.add_node("MatMul", [lhs_name, self.name], [out.name])
        return out

    # ── dot (treated as matmul for 1-D/2-D) ───────────────────────────────────

    def dot(self, other):
        return self.__matmul__(other)

    # ── indexing / slicing ────────────────────────────────────────────────────

    def __getitem__(self, key):
        """
        Support basic indexing needed for vectorized models:
          X[:, 0]     → Gather on axis=1, then Squeeze
          X[:, 0:2]   → Slice on axis=1
          p['w']      → not reached here (dicts handled separately)
        """
        ctx = _TraceContext.current()

        # Normalise to a tuple of indices
        if not isinstance(key, tuple):
            key = (key,)

        result = self
        for axis, idx in enumerate(key):
            if idx == slice(None):
                # ':' — keep this axis as-is
                continue
            elif isinstance(idx, (int, np.integer)):
                # Integer index on this axis → Gather then Squeeze
                indices_name = ctx.constant(np.array(int(idx), dtype=np.int64))
                # Gather node
                gather_out = _TracerArray._fresh(
                    tuple(d for i, d in enumerate(result.shape) if i != axis),
                    result.dtype, prefix="gather"
                )
                # ONNX Gather: output shape removes the indexed axis
                # (for 2-D tensor gathered on axis, result is 1-D)
                ctx.add_node("Gather", [result.name, indices_name], [gather_out.name],
                             axis=axis)
                result = gather_out
            elif isinstance(idx, slice):
                # Slice along this axis
                start = idx.start if idx.start is not None else 0
                stop  = idx.stop  # None means "to end" — ONNX Slice handles INT_MAX
                step  = idx.step  if idx.step  is not None else 1
                if stop is None:
                    stop = 2**31 - 1  # INT_MAX sentinel for ONNX Slice

                starts_name = ctx.constant(np.array([start], dtype=np.int64))
                ends_name   = ctx.constant(np.array([stop],  dtype=np.int64))
                axes_name   = ctx.constant(np.array([axis],  dtype=np.int64))
                steps_name  = ctx.constant(np.array([step],  dtype=np.int64))

                # Output shape: keep same dims, update the sliced axis
                new_shape = list(result.shape)
                if isinstance(new_shape[axis], int) and new_shape[axis] != -1:
                    n = new_shape[axis]
                    new_shape[axis] = len(range(start, min(stop, n), step))
                # else leave as None (dynamic)

                slice_out = _TracerArray._fresh(tuple(new_shape), result.dtype, prefix="slice")
                ctx.add_node("Slice",
                             [result.name, starts_name, ends_name, axes_name, steps_name],
                             [slice_out.name])
                result = slice_out
            else:
                raise OnnxTraceError(
                    f"Unsupported index type {type(idx).__name__!r} in ONNX tracer. "
                    "Only integer and slice indices are supported."
                )

        return result

    # ── reshape ───────────────────────────────────────────────────────────────

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        ctx = _TraceContext.current()
        shape_arr = np.array(list(shape), dtype=np.int64)
        shape_name = ctx.constant(shape_arr)
        out = _TracerArray._fresh(shape, self.dtype, prefix="reshape")
        ctx.add_node("Reshape", [self.name, shape_name], [out.name])
        return out

    # ── squeeze to scalar (Squeeze node) ──────────────────────────────────────

    def __float__(self):
        warnings.warn(
            "Attempted to cast a tracer array to float during ONNX tracing. "
            "This often means the model function branches on tensor values, "
            "which cannot be traced to ONNX. The resulting graph may be wrong.",
            OnnxTraceWarning, stacklevel=2
        )
        return 0.0  # dummy for control-flow that doesn't affect the graph

    # ── numpy ufunc protocol ───────────────────────────────────────────────────

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _UFUNC_MAP = {
            np.exp:     "Exp",
            np.log:     "Log",
            np.sqrt:    "Sqrt",
            np.tanh:    "Tanh",
            np.sin:     "Sin",
            np.cos:     "Cos",
            np.abs:     "Abs",
            np.negative: "Neg",
            np.add:     "Add",
            np.subtract: "Sub",
            np.multiply: "Mul",
            np.divide:  "Div",
            np.matmul:  "MatMul",
            np.power:   "Pow",
        }
        op_type = _UFUNC_MAP.get(ufunc)
        if op_type is None:
            raise OnnxTraceError(
                f"numpy ufunc '{ufunc.__name__}' is not supported by the ONNX tracer.\n"
                "Use save_params() / load_params() to persist model weights instead."
            )
        if method != "__call__":
            raise OnnxTraceError(f"ufunc method '{method}' is not supported by the ONNX tracer.")

        ctx = _TraceContext.current()
        input_names = []
        shape = None
        for inp in inputs:
            if isinstance(inp, _TracerArray):
                input_names.append(inp.name)
                shape = inp.shape
            else:
                input_names.append(ctx.constant(inp))
        out = _TracerArray._fresh(shape or (), np.float32)
        ctx.add_node(op_type, input_names, [out.name])
        return out

    def __array_function__(self, func, types, args, kwargs):
        return self._dispatch_numpy_func(func, *args, **kwargs)

    def _dispatch_numpy_func(self, func, *args, **kwargs):
        _FUNC_MAP = {
            np.exp:    "Exp",
            np.log:    "Log",
            np.sqrt:   "Sqrt",
            np.tanh:   "Tanh",
            np.sin:    "Sin",
            np.cos:    "Cos",
            np.abs:    "Abs",
        }
        op_type = _FUNC_MAP.get(func)
        if op_type is not None:
            return self._unary(op_type)
        raise OnnxTraceError(
            f"numpy function '{getattr(func, '__name__', func)}' is not supported "
            "by the ONNX tracer.\nUse save_params() / load_params() instead."
        )

    def __repr__(self):
        return f"_TracerArray(name={self.name!r}, shape={self.shape})"


class OnnxTraceWarning(UserWarning):
    pass


# ── numpy function overrides (module-level intercepts) ────────────────────────

class _NumpyOnnxShim:
    """
    A drop-in replacement for the ``np`` module during tracing.
    Intercepts numpy function calls and routes them through the tracer.
    """
    def __init__(self):
        self._np = np

    def __getattr__(self, name):
        return getattr(self._np, name)

    def _unary_onnx(self, op_type, x):
        if isinstance(x, _TracerArray):
            return x._unary(op_type)
        return getattr(self._np, op_type.lower())(x)

    def exp(self, x):    return self._unary_onnx("Exp", x)
    def log(self, x):    return self._unary_onnx("Log", x)
    def sqrt(self, x):   return self._unary_onnx("Sqrt", x)
    def tanh(self, x):   return self._unary_onnx("Tanh", x)
    def sin(self, x):    return self._unary_onnx("Sin", x)
    def cos(self, x):    return self._unary_onnx("Cos", x)
    def abs(self, x):    return self._unary_onnx("Abs", x)
    def arctan2(self, y, x):
        # arctan2 has no direct ONNX op; approximate via Atan(y/x) with quadrant fix
        # For simplicity, raise a clear error
        if isinstance(y, _TracerArray) or isinstance(x, _TracerArray):
            raise OnnxTraceError(
                "np.arctan2 is not directly supported by the ONNX tracer "
                "(no standard ONNX op). Refactor your model or use save_params()."
            )
        return self._np.arctan2(y, x)

    def dot(self, a, b):
        if isinstance(a, _TracerArray):
            return a.__matmul__(b)
        if isinstance(b, _TracerArray):
            return b.__rmatmul__(a)
        return self._np.dot(a, b)

    def asarray(self, x, dtype=None):
        if isinstance(x, _TracerArray):
            return x
        return self._np.asarray(x, dtype=dtype)

    def array(self, x, dtype=None):
        if isinstance(x, _TracerArray):
            return x
        return self._np.array(x, dtype=dtype)

    def clip(self, x, a_min, a_max):
        if isinstance(x, _TracerArray):
            ctx = _TraceContext.current()
            min_name = ctx.constant(float(a_min))
            max_name = ctx.constant(float(a_max))
            out = _TracerArray._fresh(x.shape, x.dtype)
            ctx.add_node("Clip", [x.name, min_name, max_name], [out.name])
            return out
        return self._np.clip(x, a_min, a_max)

    def log1p(self, x):
        # log1p(x) = log(1 + x) — compose from Add + Log
        if isinstance(x, _TracerArray):
            ctx = _TraceContext.current()
            one_name = ctx.constant(np.float32(1.0))
            add_out = _TracerArray._fresh(x.shape, x.dtype)
            ctx.add_node("Add", [x.name, one_name], [add_out.name])
            log_out = _TracerArray._fresh(add_out.shape, add_out.dtype)
            ctx.add_node("Log", [add_out.name], [log_out.name])
            return log_out
        return self._np.log1p(x)

    def squeeze(self, x, axis=None):
        if isinstance(x, _TracerArray):
            ctx = _TraceContext.current()
            new_shape = tuple(d for d in x.shape if d != 1)
            out = _TracerArray._fresh(new_shape, x.dtype, prefix="squeeze")
            if axis is not None:
                axes_name = ctx.constant(np.array([axis], dtype=np.int64))
                ctx.add_node("Squeeze", [x.name, axes_name], [out.name])
            else:
                ctx.add_node("Squeeze", [x.name], [out.name])
            return out
        return self._np.squeeze(x, axis=axis)

    def expand_dims(self, x, axis):
        if isinstance(x, _TracerArray):
            ctx = _TraceContext.current()
            new_shape = list(x.shape)
            new_shape.insert(axis, 1)
            out = _TracerArray._fresh(tuple(new_shape), x.dtype, prefix="unsqueeze")
            axes_name = ctx.constant(np.array([axis], dtype=np.int64))
            ctx.add_node("Unsqueeze", [x.name, axes_name], [out.name])
            return out
        return self._np.expand_dims(x, axis)

    def broadcast_shapes(self, *shapes):
        return self._np.broadcast_shapes(*shapes)

    def isscalar(self, x):
        if isinstance(x, _TracerArray):
            return False
        return self._np.isscalar(x)

    def finfo(self, dtype):
        return self._np.finfo(dtype)

    def random(self):
        return self._np.random

    def zeros(self, *args, **kwargs):
        return self._np.zeros(*args, **kwargs)

    def ones(self, *args, **kwargs):
        return self._np.ones(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Trace context
# ──────────────────────────────────────────────────────────────────────────────

class _TraceContext:
    """
    Thread-local singleton that accumulates ONNX nodes during a trace pass.
    """
    _instance = None

    def __init__(self):
        self.nodes = []          # list of (op_type, input_names, output_names, attrs)
        self.initializers = []   # (name, numpy_array) — constants baked in
        self._const_counter = 0

    @classmethod
    def current(cls) -> "_TraceContext":
        if cls._instance is None:
            raise RuntimeError("No active trace context. Call within a _TraceContext() block.")
        return cls._instance

    def __enter__(self):
        _TraceContext._instance = self
        _TracerArray._node_counter = 0
        return self

    def __exit__(self, *_):
        _TraceContext._instance = None

    def add_node(self, op_type, inputs, outputs, **attrs):
        self.nodes.append((op_type, inputs, outputs, attrs))

    def constant(self, value, dtype=None) -> str:
        """
        Register a numpy scalar/array as an ONNX initializer and return its name.

        If ``dtype`` is None:
          - numpy integer dtypes and Python ints are stored as int64
            (required for Gather indices, Slice params, etc.)
          - everything else is stored as float32
        """
        self._const_counter += 1
        name = f"const_{self._const_counter}"
        if dtype is not None:
            arr = np.asarray(value, dtype=dtype)
        else:
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64)
            else:
                arr = arr.astype(np.float32)
        self.initializers.append((name, arr))
        return name


# ──────────────────────────────────────────────────────────────────────────────
# Tracer: wrap parameters as _TracerArray, call f, collect nodes
# ──────────────────────────────────────────────────────────────────────────────

def _trace_model(f, p: dict, input_name: str, input_shape: tuple) -> _TraceContext:
    """
    Run a single forward pass of ``f(X_tracer, p_tracer)`` and collect the
    ONNX computation graph in a ``_TraceContext``.

    Parameters
    ----------
    f           : callable  — vectorized model function
    p           : dict      — current (learned) parameter values
    input_name  : str       — ONNX input tensor name
    input_shape : tuple     — (n_features,)  — shape of ONE input sample
                             (n_samples is symbolic; trace uses shape (None, n_features))

    Returns
    -------
    ctx         : _TraceContext with .nodes and .initializers populated
    output_name : str — the name of the traced output tensor
    """
    with _TraceContext() as ctx:
        # ── Build tracer parameter dict ────────────────────────────────────────
        p_tracer = {}
        for key, val in p.items():
            arr = np.asarray(val, dtype=np.float32)
            init_name = f"param_{key}"
            ctx.initializers.append((init_name, arr))
            p_tracer[key] = _TracerArray(init_name, arr.shape, np.float32)

        # ── Build tracer input X  ──────────────────────────────────────────────
        # shape: (batch, n_features) — use None for batch dim (dynamic)
        n_features = input_shape[0] if len(input_shape) == 1 else input_shape
        X_tracer = _TracerArray(input_name, (None, n_features), np.float32)

        # ── Call f with a shim numpy module ───────────────────────────────────
        # We patch the function's globals to intercept numpy calls.
        import types
        shim = _NumpyOnnxShim()

        # Find which numpy name the function's module uses
        fn_globals = getattr(f, "__globals__", {})
        np_aliases = [k for k, v in fn_globals.items()
                      if v is np or (isinstance(v, type(np)) and v.__name__ == "numpy")]

        # Temporarily replace np in the function's globals
        saved = {}
        for alias in np_aliases:
            saved[alias] = fn_globals[alias]
            fn_globals[alias] = shim

        try:
            result = f(X_tracer, p_tracer)
        except OnnxTraceError:
            raise
        except Exception as e:
            raise OnnxTraceError(
                f"ONNX tracing failed during forward pass: {e}\n\n"
                "This usually means the model function uses Python control flow that\n"
                "depends on tensor values, or a numpy operation not supported by the tracer.\n"
                "Try save_params() / load_params() to persist weights instead."
            ) from e
        finally:
            for alias, orig in saved.items():
                fn_globals[alias] = orig

        if not isinstance(result, _TracerArray):
            raise OnnxTraceError(
                "Model function did not return a _TracerArray after tracing — "
                "the output may be a Python scalar or numpy array, which means "
                "the function short-circuited before producing traceable ops.\n"
                "Ensure the model uses vectorized array operations throughout."
            )
        output_name = result.name

    return ctx, output_name


# ──────────────────────────────────────────────────────────────────────────────
# ONNX graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def _build_onnx_model(ctx: _TraceContext, output_name: str,
                      input_name: str, input_shape: tuple,
                      model_type: str, opset: int = 17) -> "onnx.ModelProto":
    """
    Assemble a traced ``_TraceContext`` into an ``onnx.ModelProto``.
    """
    onnx = _require_onnx()
    from onnx import helper, TensorProto, numpy_helper

    # ── Inputs ────────────────────────────────────────────────────────────────
    # Dynamic batch dimension: shape = (None, n_features)
    n_features = input_shape[0] if len(input_shape) == 1 else input_shape[-1]
    X_type = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT,
        [None, n_features]   # None = dynamic batch
    )

    # ── Outputs ───────────────────────────────────────────────────────────────
    # Classifier: (n,) probabilities; Regressor: (n,) predictions
    Y_type = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT,
        [None]   # (n_samples,)
    )

    # ── Initializers (params + constants) ─────────────────────────────────────
    # Preserve each array's dtype — integer constants (e.g. Gather indices)
    # must stay as int64; only float parameters get float32.
    initializers = []
    for name, arr in ctx.initializers:
        if np.issubdtype(arr.dtype, np.integer):
            tensor = numpy_helper.from_array(arr.astype(np.int64), name=name)
        else:
            tensor = numpy_helper.from_array(arr.astype(np.float32), name=name)
        initializers.append(tensor)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes = []
    for op_type, inputs, outputs, attrs in ctx.nodes:
        node = helper.make_node(op_type, inputs=inputs, outputs=outputs, **attrs)
        nodes.append(node)

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph = helper.make_graph(
        nodes,
        name="LambdaMLGraph",
        inputs=[X_type],
        outputs=[Y_type],
        initializer=initializers,
    )

    # ── Model metadata ─────────────────────────────────────────────────────────
    model_proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model_proto.doc_string = (
        f"LambdaML {model_type} exported via to_onnx(). "
        f"Parameters are embedded as initializers."
    )
    model_proto.domain = "lambdaml"
    model_proto.model_version = 1

    # Add custom metadata
    meta = model_proto.metadata_props
    entry = meta.add()
    entry.key = "lambdaml_model_type"
    entry.value = model_type

    # Use IR version 8 for broad onnxruntime compatibility
    model_proto.ir_version = 8

    onnx.checker.check_model(model_proto)
    return model_proto


# ──────────────────────────────────────────────────────────────────────────────
# Public API: to_onnx
# ──────────────────────────────────────────────────────────────────────────────

def to_onnx(
    model: "_LambdaBaseModel",
    path: str | None = None,
    *,
    input_shape: tuple,
    input_name: str = "X",
    opset: int = 17,
) -> "onnx.ModelProto":
    """
    Export a fitted LambdaML model to ONNX format.

    The model **must** have been created with ``vectorized=True``.
    The function ``f(X, p)`` must use standard numpy operations so the
    tracer can reconstruct the computation graph.

    Parameters
    ----------
    model       : LambdaClassifierModel or LambdaRegressorModel
                  The fitted model to export.
    path        : str or None
                  File path to write the ``.onnx`` file.
                  If None, the model proto is returned but not written.
    input_shape : tuple
                  Shape of a *single* input sample, e.g. ``(2,)`` for 2 features.
                  The batch dimension is added automatically (dynamic).
    input_name  : str
                  Name of the ONNX input tensor (default ``"X"``).
    opset       : int
                  ONNX opset version (default 17, supported by onnxruntime >= 1.14).

    Returns
    -------
    onnx.ModelProto
        The ONNX model. Also written to ``path`` if provided.

    Raises
    ------
    OnnxTraceError
        If the model is not vectorized or the function uses unsupported ops.
    ImportError
        If ``onnx`` is not installed.

    Examples
    --------
    >>> def logistic_v(X, p):
    ...     return 1 / (1 + np.exp(-(X @ p['w'] + p['b'])))
    >>> model = LambdaClassifierModel(f=logistic_v, p={'w': np.zeros(2), 'b': 0.0},
    ...                               vectorized=True)
    >>> model.fit(X_train, Y_train, n_iter=200, lr=0.01)
    >>> proto = model.to_onnx('model.onnx', input_shape=(2,))
    """
    onnx_mod = _require_onnx()

    if not model.vectorized:
        raise OnnxTraceError(
            "to_onnx() requires vectorized=True on the model.\n\n"
            "Rewrite your model function to accept the full X matrix:\n\n"
            "  # Non-vectorized (won't trace):\n"
            "  def f(x, p):  # x is a single sample (1-D)\n"
            "      return ...\n\n"
            "  # Vectorized (traceable):\n"
            "  def f(X, p):  # X is (n_samples, n_features)\n"
            "      return ...  # must return (n_samples,) array\n\n"
            "Then recreate your model with vectorized=True.\n"
            "Alternatively, use save_params() / load_params() (always works)."
        )

    model_type = type(model).__name__

    # ── Trace ─────────────────────────────────────────────────────────────────
    ctx, output_name = _trace_model(model.f, model.p, input_name, input_shape)

    # ── Assemble ──────────────────────────────────────────────────────────────
    proto = _build_onnx_model(ctx, output_name, input_name, input_shape,
                              model_type=model_type, opset=opset)

    # ── Write ─────────────────────────────────────────────────────────────────
    if path is not None:
        with open(path, "wb") as fh:
            fh.write(proto.SerializeToString())

    return proto


# ──────────────────────────────────────────────────────────────────────────────
# Public API: from_onnx / predict_onnx
# ──────────────────────────────────────────────────────────────────────────────

def from_onnx(path: str) -> "onnxruntime.InferenceSession":
    """
    Load an ONNX model exported by ``to_onnx()`` and return an
    ``onnxruntime.InferenceSession`` ready for inference.

    Parameters
    ----------
    path : str — path to the ``.onnx`` file.

    Returns
    -------
    onnxruntime.InferenceSession

    Examples
    --------
    >>> sess = from_onnx('model.onnx')
    >>> input_name = sess.get_inputs()[0].name
    >>> probs = sess.run(None, {input_name: X_test.astype(np.float32)})[0]
    """
    rt = _require_onnxruntime()
    return rt.InferenceSession(path)


def predict_onnx(
    path_or_session,
    X: np.ndarray,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Run inference on ``X`` using an ONNX model.

    Convenience wrapper around ``onnxruntime.InferenceSession.run()``.

    Parameters
    ----------
    path_or_session : str or onnxruntime.InferenceSession
        Path to an ``.onnx`` file, or an already-loaded session.
    X               : array-like, shape (n_samples, n_features)
        Input data. Converted to float32 automatically.
    threshold       : float or None
        If provided, binarise the output at this threshold (for classifiers).
        None = return raw probabilities / predictions.

    Returns
    -------
    np.ndarray — shape (n_samples,)
        Predicted probabilities (classifier) or values (regressor),
        or binary labels if ``threshold`` is set.

    Examples
    --------
    >>> probs = predict_onnx('model.onnx', X_test)
    >>> labels = predict_onnx('model.onnx', X_test, threshold=0.5)
    """
    rt = _require_onnxruntime()

    if isinstance(path_or_session, str):
        sess = rt.InferenceSession(path_or_session)
    else:
        sess = path_or_session

    X_f32 = np.asarray(X, dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: X_f32})[0].astype(np.float64)

    if threshold is not None:
        return (output >= threshold).astype(int)
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Public API: save_params / load_params  (always works, no tracing required)
# ──────────────────────────────────────────────────────────────────────────────

def save_params(model: "_LambdaBaseModel", path: str, **metadata) -> None:
    """
    Save model parameters to a compressed NumPy ``.npz`` file.

    This approach always works — no ONNX tracing, no vectorized requirement.
    On load you must reconstruct the model (same ``f`` and initial ``p``
    structure) and call ``load_params()`` to restore weights.

    Parameters
    ----------
    model    : LambdaClassifierModel or LambdaRegressorModel
    path     : str — file path (will add ``.npz`` if not present)
    **metadata : extra scalar values to store alongside the parameters
                 (e.g. ``model_type='classifier'``).

    Examples
    --------
    >>> model.save_params('weights.npz', model_type='classifier', n_features=2)
    >>> # Restore:
    >>> model2.load_params('weights.npz')
    """
    arrays = {}
    for k, v in model.p.items():
        arr = np.asarray(v, dtype=float)
        arrays[f"param_{k}"] = arr

    # Store scalar parameters that can be round-tripped through numpy
    arrays["__param_keys__"] = np.array(list(model.p.keys()))
    arrays["__model_type__"] = np.array([type(model).__name__])

    for mk, mv in metadata.items():
        arrays[f"__meta_{mk}__"] = np.array([mv])

    np.savez_compressed(path, **arrays)


def load_params(model: "_LambdaBaseModel", path: str) -> "_LambdaBaseModel":
    """
    Load parameters from a ``.npz`` file saved by ``save_params()`` into
    an existing model instance.

    The model must have been constructed with the same parameter keys as
    the saved file.  Scalar parameters are restored as Python floats;
    array parameters are restored as numpy float64 arrays.

    Parameters
    ----------
    model : LambdaClassifierModel or LambdaRegressorModel
            A freshly-constructed model (same ``f`` and ``p`` structure).
    path  : str — path to the ``.npz`` file.

    Returns
    -------
    model : the same model instance, with parameters updated in-place.

    Examples
    --------
    >>> model = LambdaClassifierModel(f=logistic_v,
    ...                               p={'w': np.zeros(2), 'b': 0.0},
    ...                               vectorized=True)
    >>> model.load_params('weights.npz')
    >>> preds = model.predict(X_test)
    """
    data = np.load(path, allow_pickle=False)
    keys = data["__param_keys__"].tolist()

    for k in keys:
        arr = data[f"param_{k}"]
        # Restore as scalar float if the saved array was 0-D or shape ()
        if arr.ndim == 0 or arr.shape == ():
            model.p[k] = float(arr)
        else:
            model.p[k] = arr.astype(float)

    return model
