"""
lambda_utils.py
---------------
Numerical differentiation and loss/regularization utilities for LambdaML.

The core idea: given any black-box scalar function f(θ) we can estimate ∂f/∂θ
without knowing f's analytic form. This is called *numerical* or *finite-difference*
differentiation. The original library used a single symmetric (central) difference.
This version expands that to six methods plus Richardson extrapolation.

Terminology note you asked about
---------------------------------
The term you were reaching for is "finite-difference approximation" or sometimes
"numerical differentiation". The word "empirical" is occasionally used informally
but the standard vocabulary is:

  • Forward difference     f'(x) ≈ [f(x+h) - f(x)] / h                O(h)
  • Backward difference    f'(x) ≈ [f(x) - f(x-h)] / h                O(h)
  • Central difference     f'(x) ≈ [f(x+h) - f(x-h)] / (2h)           O(h²)  ← original
  • Five-point stencil     f'(x) ≈ [-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)] / (12h)  O(h⁴)
  • Complex-step           f'(x) ≈ Im[f(x+ih)] / h                     O(h²) but no cancellation error
  • Richardson extrapolation  Combines two central estimates to cancel O(h²) term → O(h⁴)

Is numerical backprop tractable?
---------------------------------
YES — with caveats.
• Cost: each scalar parameter requires 1-2 extra forward passes.  For a model with
  n_params parameters the gradient step costs O(n_params) forward passes, vs O(1)
  for analytic backprop.  For small models (hundreds of params) this is fine.
  For large nets (millions of params) it becomes prohibitive.
• Accuracy: central difference and higher-order methods give excellent gradients
  for smooth functions.  The complex-step method is the most accurate of all
  (no catastrophic cancellation).
• Use-cases: prototype activation functions, non-differentiable custom losses,
  physics-based forward models, any exotic f you don't want to differentiate by hand.
"""

import numpy as np
from collections.abc import Iterable


# ──────────────────────────────────────────────────────────────────────────────
# Epsilon defaults: choose h carefully.
#   Too small → catastrophic cancellation (float rounding kills the signal).
#   Too large → truncation error (Taylor expansion breaks down).
#   Rule of thumb for central difference: h ≈ ε^(1/3) where ε is machine eps.
# ──────────────────────────────────────────────────────────────────────────────
_EPS_MACHINE = np.finfo(np.float64).eps          # ≈ 2.2e-16
_H_CENTRAL   = _EPS_MACHINE ** (1/3)             # ≈ 6e-6  — optimal for O(h²)
_H_FIVE_PT   = _EPS_MACHINE ** (1/5)             # ≈ 1e-3  — optimal for O(h⁴)
_H_FORWARD   = _EPS_MACHINE ** (1/2)             # ≈ 1.5e-8 — optimal for O(h)
_H_COMPLEX   = 1e-20                             # can be extremely small — no cancellation


class DiffMethod:
    """
    Enumeration of available finite-difference methods.

    FORWARD      — one extra f-eval, O(h) accuracy (fast, low accuracy)
    BACKWARD     — one extra f-eval, O(h) accuracy (fast, low accuracy)
    CENTRAL      — two extra f-evals, O(h²) accuracy (default, good balance)
    FIVE_POINT   — four extra f-evals, O(h⁴) accuracy (expensive, high accuracy)
    COMPLEX_STEP — one extra f-eval (complex), O(h²) accuracy *without* cancellation;
                   requires f to work with complex inputs (pure numpy usually does)
    RICHARDSON   — two central estimates at h and h/2, extrapolated to O(h⁴);
                   four extra f-evals, very accurate, works even if f can't take complex
    """
    FORWARD      = "forward"
    BACKWARD     = "backward"
    CENTRAL      = "central"
    FIVE_POINT   = "five_point"
    COMPLEX_STEP = "complex_step"
    RICHARDSON   = "richardson"


class NumericalDiff:
    """
    Stateless collection of numerical derivative methods for a scalar function.
    All methods accept f: R → R and a point x: float.
    """

    @staticmethod
    def forward(f, x, h=_H_FORWARD):
        """Forward difference: O(h) accuracy, 1 extra f-eval."""
        return (f(x + h) - f(x)) / h

    @staticmethod
    def backward(f, x, h=_H_FORWARD):
        """Backward difference: O(h) accuracy, 1 extra f-eval."""
        return (f(x) - f(x - h)) / h

    @staticmethod
    def central(f, x, h=_H_CENTRAL):
        """Central (symmetric) difference: O(h²) accuracy, 2 extra f-evals.
        This is what the original LambdaML used (incorrectly with float16 eps)."""
        return (f(x + h) - f(x - h)) / (2.0 * h)

    @staticmethod
    def five_point(f, x, h=_H_FIVE_PT):
        """Five-point stencil: O(h⁴) accuracy, 4 extra f-evals.
        Excellent for smooth functions."""
        return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12.0 * h)

    @staticmethod
    def complex_step(f, x, h=_H_COMPLEX):
        """Complex-step method: O(h²) accuracy, NO cancellation error.
        Most accurate method when f supports complex inputs (all pure-numpy fns do).
        Formula: f'(x) ≈ Im[f(x + ih)] / h"""
        return np.imag(f(x + 1j * h)) / h

    @staticmethod
    def richardson(f, x, h=_H_CENTRAL):
        """Richardson extrapolation: combines two central estimates to get O(h⁴).
        Works when f cannot accept complex inputs.
        d1 = central(f, x, h); d2 = central(f, x, h/2)
        extrapolated = (4*d2 - d1) / 3"""
        d1 = NumericalDiff.central(f, x, h)
        d2 = NumericalDiff.central(f, x, h / 2.0)
        return (4.0 * d2 - d1) / 3.0

    # Dispatch map
    _methods = {
        DiffMethod.FORWARD:      forward.__func__,
        DiffMethod.BACKWARD:     backward.__func__,
        DiffMethod.CENTRAL:      central.__func__,
        DiffMethod.FIVE_POINT:   five_point.__func__,
        DiffMethod.COMPLEX_STEP: complex_step.__func__,
        DiffMethod.RICHARDSON:   richardson.__func__,
    }

    @classmethod
    def differentiate(cls, f, x, method=DiffMethod.CENTRAL, h=None):
        """
        Differentiate f at x using the given method.
        If h is None, the method's default step size is used.
        """
        fn = cls._methods[method]
        if h is not None:
            return fn(f, x, h)
        return fn(f, x)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient computation over the parameter dict
# ──────────────────────────────────────────────────────────────────────────────

class GradientComputer:
    """
    Computes numerical gradients of an objective function with respect to a
    parameter dictionary.  Supports scalar params and numpy array params.
    """

    def __init__(self, method=DiffMethod.CENTRAL, h=None):
        self.method = method
        self.h = h

    def _scalar_grad(self, f_of_z, z):
        """Gradient of f (scalar → scalar) at scalar z."""
        return NumericalDiff.differentiate(f_of_z, z, method=self.method, h=self.h)

    def _array_grad(self, f_factory, arr):
        """
        Gradient of f with respect to a numpy array parameter.
        f_factory(i) returns a scalar→scalar function that perturbs arr[i].
        Returns a numpy array of the same shape as arr.
        """
        grad = np.zeros_like(arr, dtype=np.float64)
        # Flatten for iteration; handles arbitrarily shaped arrays
        flat_arr = arr.ravel()
        flat_grad = grad.ravel()
        for i in range(len(flat_arr)):
            flat_grad[i] = NumericalDiff.differentiate(
                f_factory(i), flat_arr[i], method=self.method, h=self.h
            )
        return flat_grad.reshape(arr.shape)

    def compute(self, objective, p, key, *args, **kwargs):
        """
        Compute ∂(objective)/∂p[key] numerically.

        Parameters
        ----------
        objective : callable(*args, **kwargs) given modified p  →  scalar
            Must accept p as first positional argument.
        p         : dict of parameters
        key       : which parameter to differentiate w.r.t.
        *args, **kwargs : extra arguments forwarded to objective after p

        Returns
        -------
        Gradient with the same shape/type as p[key].
        """
        val = p[key]

        if isinstance(val, np.ndarray):
            def make_perturbed(idx):
                def f_of_z(z):
                    p_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                              for k, v in p.items()}
                    flat = p_copy[key].ravel().copy().astype(complex if self.method == DiffMethod.COMPLEX_STEP else float)
                    flat[idx] = z
                    p_copy[key] = flat.reshape(p_copy[key].shape)
                    return objective(p_copy, *args, **kwargs)
                return f_of_z
            return self._array_grad(make_perturbed, val)

        elif isinstance(val, Iterable):
            # Generic iterable (list, etc.) — convert to array first
            arr = np.array(val, dtype=float)
            def make_perturbed(idx):
                def f_of_z(z):
                    p_copy = dict(p)
                    new_arr = arr.copy()
                    new_arr[idx] = z
                    p_copy[key] = new_arr
                    return objective(p_copy, *args, **kwargs)
                return f_of_z
            return self._array_grad(make_perturbed, arr)

        else:
            # Scalar
            def f_of_z(z):
                p_copy = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in p.items()}
                p_copy[key] = z
                return objective(p_copy, *args, **kwargs)

            if self.method == DiffMethod.COMPLEX_STEP:
                # For complex step on a scalar we need complex z
                return NumericalDiff.differentiate(f_of_z, complex(val), method=self.method, h=self.h).real
            return NumericalDiff.differentiate(f_of_z, float(val), method=self.method, h=self.h)


# ──────────────────────────────────────────────────────────────────────────────
# Regularization
# ──────────────────────────────────────────────────────────────────────────────

class Regularization:
    """
    Standard regularization penalties on a parameter dictionary.

    NOTE — bug fix from original:
    The original l1_regularization summed raw values (not absolute values), so
    negative parameters would *decrease* the penalty, which is wrong.
    L1 should penalize |θ|, not θ.
    """

    @staticmethod
    def l1(p, skip_keys=None):
        """
        L1 penalty: Σ |θᵢ|  (LASSO-style; encourages sparsity).
        Fixed from original: uses |θ| not θ (negative params used to reduce penalty).

        For complex-step compatibility we use np.sqrt(v*v) instead of np.abs(v),
        which is differentiable and propagates imaginary parts correctly.
        (np.abs kills imaginary components.)
        """
        skip_keys = skip_keys or set()
        total = 0.0
        for k, v in p.items():
            if k in skip_keys:
                continue
            if isinstance(v, np.ndarray):
                total += np.sqrt(v * v + 1e-30).sum()   # smooth |v|, complex-safe
            else:
                total += np.sqrt(v * v + 1e-30)          # smooth |v|, complex-safe
        return total

    @staticmethod
    def l2(p, skip_keys=None):
        """
        L2 penalty: Σ θᵢ²  (Ridge-style; penalizes large weights).
        Complex-step compatible: penalty is computed on the real part of params
        (the imaginary perturbation is O(h²) and correctly propagates through
        the addition to give the right gradient via Im[f(x+ih)]/h).
        """
        skip_keys = skip_keys or set()
        total = 0.0
        for k, v in p.items():
            if k in skip_keys:
                continue
            if isinstance(v, np.ndarray):
                # Use v*v (not np.square) to preserve complex imaginary parts
                total += (v * v).sum()
            else:
                total += v * v
        return total

    @staticmethod
    def elastic_net(p, l1_ratio=0.5, skip_keys=None):
        """
        Elastic-net: α·L1 + (1-α)·L2 combined penalty.
        l1_ratio=1 → pure L1; l1_ratio=0 → pure L2.
        """
        return (l1_ratio * Regularization.l1(p, skip_keys) +
                (1 - l1_ratio) * Regularization.l2(p, skip_keys))


# ──────────────────────────────────────────────────────────────────────────────
# Objective / loss functions
# ──────────────────────────────────────────────────────────────────────────────

class LossFunctions:
    """
    Vectorized loss functions that consume (y_pred array, y_true array) → scalar.
    All return *negative* quantities for maximization-style objectives, or
    positive quantities for minimization. The model's optimizer always *minimizes*
    the returned loss.
    """

    @staticmethod
    def binary_cross_entropy(y_pred, y_true, eps=1e-12):
        """
        Binary cross-entropy (log-loss).  Equivalent to negative log-likelihood.
        -(y·log(p) + (1-y)·log(1-p))
        """
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def mse(y_pred, y_true):
        """Mean squared error — for regression."""
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def mae(y_pred, y_true):
        """Mean absolute error — more robust to outliers."""
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def huber(y_pred, y_true, delta=1.0):
        """
        Huber loss — quadratic for |error| < delta, linear beyond.
        Robust alternative to MSE.

        Note: uses np.where which is NOT complex-step-compatible.
        Use pseudo_huber instead if you're using DiffMethod.COMPLEX_STEP.
        """
        r = np.abs(np.real(y_pred) - y_true)   # real part for condition
        return np.mean(np.where(r <= delta,
                                0.5 * r ** 2,
                                delta * (r - 0.5 * delta)))

    @staticmethod
    def pseudo_huber(y_pred, y_true, delta=1.0):
        """
        Smooth (differentiable) Huber loss using the Pseudo-Huber formulation.
        Fully complex-step-compatible — safe with all diff methods.
        Formula: δ² · (√(1 + (r/δ)²) - 1)

        Important: no .real call here — the imaginary part must be preserved
        for complex-step differentiation to work correctly.
        """
        r = y_pred - y_true
        return np.mean(delta ** 2 * (np.sqrt(1 + (r / delta) ** 2) - 1))

    @staticmethod
    def log_likelihood(y_pred, y_true, eps=1e-12):
        """
        Raw log-likelihood (positive; maximized).  Negate to get a loss.
        """
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ──────────────────────────────────────────────────────────────────────────────
# Learning-rate schedules
# ──────────────────────────────────────────────────────────────────────────────

class LRSchedule:
    """Learning rate schedule callables: (initial_lr, epoch) → current_lr."""

    @staticmethod
    def constant(lr):
        return lambda lr0, t: lr0

    @staticmethod
    def step_decay(drop=0.5, epochs_drop=10):
        """Halve LR every epochs_drop iterations."""
        return lambda lr0, t: lr0 * (drop ** (t // epochs_drop))

    @staticmethod
    def exponential_decay(k=0.01):
        """lr = lr0 * exp(-k*t)."""
        return lambda lr0, t: lr0 * np.exp(-k * t)

    @staticmethod
    def cosine_annealing(T_max=100):
        """lr = lr0 * 0.5 * (1 + cos(π*t/T_max))."""
        return lambda lr0, t: lr0 * 0.5 * (1 + np.cos(np.pi * t / T_max))

    @staticmethod
    def warmup_cosine(warmup=10, T_max=100):
        """Linear warmup then cosine annealing."""
        def schedule(lr0, t):
            if t < warmup:
                return lr0 * (t + 1) / warmup
            return lr0 * 0.5 * (1 + np.cos(np.pi * (t - warmup) / (T_max - warmup)))
        return schedule
