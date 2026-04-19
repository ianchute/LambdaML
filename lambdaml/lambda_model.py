"""
lambda_model.py
---------------
LambdaML core model classes.

Design philosophy
-----------------
You provide:
  f(x, p)  →  scalar prediction (probability or real value)
  p        →  dict of parameters (scalars or numpy arrays)

LambdaML provides:
  • Numerical gradient computation (6 methods; no analytical derivatives needed)
  • Vectorized batch/mini-batch objective evaluation
  • Multiple optimizers: SGD, Momentum, Adam, RMSProp
  • L1 / L2 / Elastic-Net regularization (with corrected L1 formula)
  • Learning rate scheduling
  • Early stopping with patience
  • Convergence history tracking
  • LambdaClassifierModel  (binary cross-entropy objective, sigmoid output expected)
  • LambdaRegressorModel   (MSE / MAE / Huber objective)

Backprop correctness notes vs the original
------------------------------------------
Original bug 1 — epsilon was np.finfo(np.float16).eps ≈ 0.001.
  For central difference this caused massive truncation error.  Fixed: use
  float64 machine epsilon and the optimal step size for each method.

Original bug 2 — L1 regularization summed raw parameter values, so negative
  parameters reduced the penalty (wrong). Fixed: sum |θ|.

Original bug 3 — gradient was added directly to p[key] but this means we
  were doing *gradient ascent on the log-likelihood*, which is correct for MLE
  but the sign convention was implicit and fragile. Now the model minimizes a
  loss explicitly, making the direction of optimization unambiguous.

Original bug 4 — For array parameters, the gradient loop created a closure
  over a mutable loop variable `ix`, so all closures captured the *last* index.
  (Classic Python closure-in-loop bug.) Fixed by using a proper factory function.

Performance improvements (v1.0.4)
----------------------------------
• Cached skip set — _reg_skip_keys computed once at __init__, not per epoch.
• Cached optimizer step fn — resolved once at __init__ to avoid per-step string
  comparisons inside the hot parameter loop.
• Single-pass regularization — _reg_penalty iterates p once regardless of
  whether L1, L2, or both are active (was two separate passes before).
• Default eval_every raised to 10 — avoids a redundant full forward pass on
  90% of epochs out of the box.
• Parallel gradient computation — n_jobs parameter on fit() dispatches each
  parameter's gradient to a separate worker via joblib (opt-in).
"""

import numpy as np
try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    from joblib import Parallel, delayed as _jl_delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from .lambda_utils import (
    GradientComputer, DiffMethod,
    Regularization, LossFunctions, LRSchedule
)
from . import lambda_onnx as _onnx_module


# ──────────────────────────────────────────────────────────────────────────────
# Optimizers
# ──────────────────────────────────────────────────────────────────────────────

def _make_optimizer_step_fn(optimizer_name, momentum, beta1, beta2, adam_eps):
    """
    Return a pre-bound step function for the chosen optimizer.

    Resolves the optimizer string *once* at model construction time so that the
    hot per-parameter loop inside fit() does not repeat string comparisons every
    epoch for every parameter.

    Returns
    -------
    fn(p, grads, state, lr) → None   (updates p in-place)
    """
    if optimizer_name == Optimizer.SGD:
        def _step(p, grads, state, lr):
            state.t += 1
            for key in p:
                p[key] -= lr * grads[key]

    elif optimizer_name == Optimizer.MOMENTUM:
        def _step(p, grads, state, lr):
            state.t += 1
            for key in p:
                state.m[key] = momentum * state.m[key] + lr * grads[key]
                p[key] -= state.m[key]

    elif optimizer_name == Optimizer.RMSPROP:
        def _step(p, grads, state, lr):
            state.t += 1
            for key in p:
                state.v[key] = beta2 * state.v[key] + (1 - beta2) * grads[key] ** 2
                p[key] -= lr * grads[key] / (np.sqrt(state.v[key]) + adam_eps)

    elif optimizer_name == Optimizer.ADAM:
        def _step(p, grads, state, lr):
            state.t += 1
            t = state.t
            for key in p:
                g = grads[key]
                state.m[key] = beta1 * state.m[key] + (1 - beta1) * g
                state.v[key] = beta2 * state.v[key] + (1 - beta2) * g ** 2
                m_hat = state.m[key] / (1 - beta1 ** t)
                v_hat = state.v[key] / (1 - beta2 ** t)
                p[key] -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. "
                         f"Choose from: sgd, momentum, rmsprop, adam")
    return _step


class _OptimizerState:
    """Holds per-parameter optimizer state (momentum buffers, Adam moments, etc.)"""

    def __init__(self, p):
        self.m = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0)
                  for k, v in p.items()}
        self.v = {k: (np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0)
                  for k, v in p.items()}
        self.t = 0   # step counter for Adam bias correction


class Optimizer:
    """
    Optimizer dispatch.  All apply methods update p in-place and return p.
    """

    SGD       = "sgd"
    MOMENTUM  = "momentum"
    RMSPROP   = "rmsprop"
    ADAM      = "adam"

    @staticmethod
    def step(optimizer, p, grads, state, lr,
             momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Apply one gradient step.

        Parameters
        ----------
        optimizer : str  — one of Optimizer.{SGD, MOMENTUM, RMSPROP, ADAM}
        p         : dict — parameters, updated in-place
        grads     : dict — gradient for each key (same keys as p)
        state     : _OptimizerState — mutable momentum/Adam state
        lr        : float — current learning rate
        """
        state.t += 1

        for key in p:
            g = grads[key]

            if optimizer == Optimizer.SGD:
                p[key] -= lr * g

            elif optimizer == Optimizer.MOMENTUM:
                state.m[key] = momentum * state.m[key] + lr * g
                p[key] -= state.m[key]

            elif optimizer == Optimizer.RMSPROP:
                state.v[key] = beta2 * state.v[key] + (1 - beta2) * g ** 2
                p[key] -= lr * g / (np.sqrt(state.v[key]) + epsilon)

            elif optimizer == Optimizer.ADAM:
                state.m[key] = beta1 * state.m[key] + (1 - beta1) * g
                state.v[key] = beta2 * state.v[key] + (1 - beta2) * g ** 2
                m_hat = state.m[key] / (1 - beta1 ** state.t)
                v_hat = state.v[key] / (1 - beta2 ** state.t)
                p[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

            else:
                raise ValueError(f"Unknown optimizer: {optimizer!r}. "
                                 f"Choose from: sgd, momentum, rmsprop, adam")

        return p


# ──────────────────────────────────────────────────────────────────────────────
# Base model
# ──────────────────────────────────────────────────────────────────────────────

class _LambdaBaseModel:
    """
    Internal base class shared by classifier and regressor.
    """

    def __init__(
        self,
        f,
        p,
        *,
        diff_method=DiffMethod.CENTRAL,
        diff_h=None,
        l1_factor=0.0,
        l2_factor=0.01,
        l1_ratio=0.5,           # only used when both l1 and l2 are nonzero
        regularize_bias=False,  # whether to regularize bias ('b*') params
        optimizer=Optimizer.ADAM,
        momentum=0.9,
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        lr_schedule=None,
        vectorized=False,       # if True, f(X, p) must accept the full X matrix
        n_jobs=1,               # parallel gradient workers (requires joblib)
    ):
        self.f             = f
        self.p             = {k: (np.array(v, dtype=float) if isinstance(v, (list, tuple))
                                  else v)
                              for k, v in p.items()}
        self.diff_method   = diff_method
        self.diff_h        = diff_h
        self.l1_factor     = l1_factor
        self.l2_factor     = l2_factor
        self.l1_ratio      = l1_ratio
        self.regularize_bias = regularize_bias
        self.optimizer_name = optimizer
        self.momentum      = momentum
        self.beta1         = beta1
        self.beta2         = beta2
        self.adam_eps      = adam_eps
        self.lr_schedule   = lr_schedule or (lambda lr0, t: lr0)  # constant
        self.vectorized    = vectorized
        self.n_jobs        = n_jobs

        self._gc = GradientComputer(method=diff_method, h=diff_h)
        self._opt_state = _OptimizerState(self.p)

        # ── Cache the regularization skip set once (avoids rebuilding every epoch) ──
        if regularize_bias:
            self._reg_skip_keys = frozenset()
        else:
            self._reg_skip_keys = frozenset(k for k in self.p if k.startswith('b'))

        # ── Pre-bind the optimizer step function (avoids string comparison per step) ──
        self._opt_step_fn = _make_optimizer_step_fn(
            optimizer, momentum, beta1, beta2, adam_eps
        )

        # History
        self.loss_history = []

    # ------------------------------------------------------------------
    # Regularization penalty (added to loss, not subtracted from log-lik)
    # ------------------------------------------------------------------

    def _reg_penalty(self, p):
        """
        Single-pass regularization penalty.

        Previously the code called Regularization.l1() and Regularization.l2()
        as separate methods, each iterating over all parameters — two full passes
        when both L1 and L2 were active.  This version completes both sums in a
        single loop, and uses the skip set that was cached at __init__ time.
        """
        l1f = self.l1_factor
        l2f = self.l2_factor
        if l1f == 0.0 and l2f == 0.0:
            return 0.0

        skip = self._reg_skip_keys
        l1_sum = 0.0
        l2_sum = 0.0

        for k, v in p.items():
            if k in skip:
                continue
            if isinstance(v, np.ndarray):
                if l1f:
                    l1_sum += np.sqrt(v * v + 1e-30).sum()   # smooth |v|, complex-safe
                if l2f:
                    l2_sum += (v * v).sum()
            else:
                if l1f:
                    l1_sum += np.sqrt(v * v + 1e-30)
                if l2f:
                    l2_sum += v * v

        return l1f * l1_sum + l2f * l2_sum

    # ------------------------------------------------------------------
    # Prediction (subclasses implement _predict_batch)
    # ------------------------------------------------------------------

    def _predict_batch(self, X, p, show_progress=False, progress_desc="Predicting"):
        """
        Return array of predictions for all rows in X.

        If self.vectorized=True, calls f(X, p) once (X is the full matrix).
        Otherwise loops per sample, optionally showing a tqdm progress bar.

        Parameters
        ----------
        show_progress : bool  — show a per-sample tqdm bar (ignored when vectorized)
        progress_desc : str   — label shown on the bar
        """
        if self.vectorized:
            return np.asarray(self.f(X, p), dtype=float)

        if show_progress and _TQDM_AVAILABLE:
            samples = _tqdm(X, desc=progress_desc, unit="sample", leave=False)
        else:
            samples = X
        return np.array([self.f(x, p) for x in samples])

    # ------------------------------------------------------------------
    # Loss (subclasses override _loss_fn)
    # ------------------------------------------------------------------

    def _loss_fn(self, y_pred, y_true):
        raise NotImplementedError

    def _objective(self, p, X, Y):
        """Full objective = data loss + regularization penalty."""
        y_pred = self._predict_batch(X, p)
        return self._loss_fn(y_pred, Y) + self._reg_penalty(p)

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _compute_gradients(self, X, Y):
        """
        Compute gradients for all parameters.

        When n_jobs != 1 and joblib is available, each parameter's gradient is
        computed in a separate worker process, giving near-linear speedup on
        multi-core machines for models with many parameters.

        Note on parallelism and modify-and-restore
        ------------------------------------------
        The modify-and-restore pattern in GradientComputer mutates p in-place,
        so it is NOT safe to parallelise across elements of the *same* parameter.
        However, gradients for *different* parameters are independent — each
        worker receives its own deep copy of p for its key, applies
        modify-and-restore locally, and returns the gradient scalar/array.
        The main process p is never touched by workers.
        """
        if self.n_jobs != 1 and _JOBLIB_AVAILABLE and len(self.p) > 1:
            return self._compute_gradients_parallel(X, Y)
        return self._compute_gradients_sequential(X, Y)

    def _compute_gradients_sequential(self, X, Y):
        grads = {}
        for key in self.p:
            grads[key] = self._gc.compute(self._objective, self.p, key, X, Y)
        return grads

    def _compute_gradients_parallel(self, X, Y):
        """
        Parallel gradient computation via joblib.

        Each worker gets an independent copy of p so that modify-and-restore
        mutations in one worker do not affect another.  Results are collected
        and merged back into a single grads dict.
        """
        import copy

        def _grad_for_key(key, p_snapshot, X, Y):
            # Each worker operates on its own isolated parameter dict
            gc = GradientComputer(method=self._gc.method, h=self._gc.h)
            return key, gc.compute(self._objective_from_snapshot, p_snapshot, key, X, Y)

        # Take a consistent snapshot of p before dispatching workers
        p_snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                      for k, v in self.p.items()}

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            _jl_delayed(_grad_for_key)(key, copy.deepcopy(p_snapshot), X, Y)
            for key in self.p
        )
        return dict(results)

    def _objective_from_snapshot(self, p_snap, X, Y):
        """
        Objective that uses a snapshot p dict instead of self.p.
        Used by parallel workers so they don't share mutable state.
        """
        if self.vectorized:
            y_pred = np.asarray(self.f(X, p_snap), dtype=float)
        else:
            y_pred = np.array([self.f(x, p_snap) for x in X])
        return self._loss_fn(y_pred, Y) + self._reg_penalty(p_snap)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        Y,
        *,
        n_iter=100,
        lr=0.01,
        batch_size=None,
        early_stopping=False,
        patience=10,
        tol=1e-6,
        verbose=False,
        validation_data=None,
        progress_bar=True,
        eval_every=10,
    ):
        """
        Fit parameters by minimizing the objective.

        Parameters
        ----------
        X            : array-like, shape (n_samples, n_features)
        Y            : array-like, shape (n_samples,)
        n_iter       : int  — maximum number of gradient steps
        lr           : float — initial learning rate
        batch_size   : int or None — mini-batch size; None = full batch
        early_stopping : bool — stop if loss hasn't improved by tol for patience steps
        patience     : int — early stopping patience (epochs)
        tol          : float — minimum improvement threshold
        verbose      : bool — print loss every eval_every iterations
        validation_data : tuple (X_val, Y_val) or None
        progress_bar : bool — show a tqdm progress bar over epochs (default True)
        eval_every   : int  — compute and log loss every N epochs (default 10).
                       Each loss evaluation is a full forward pass over the dataset,
                       so reducing this frequency is a free speedup.  Set to 1 to
                       restore the old behaviour.

        Returns
        -------
        self

        Speed tips
        ----------
        The main cost is O(n_params * diff_evals * n_samples) per epoch.
        To speed up training:
          • n_jobs=-1 on the model — parallelises gradient computation across
            parameters using all CPU cores (requires joblib).
          • vectorized=True on the model + rewrite f(X, p) to accept the full
            matrix — eliminates the Python sample loop entirely.
          • eval_every=50 or higher — each loss eval is a full forward pass;
            you rarely need it every epoch.
          • batch_size=N — reduces samples used per gradient step.
          • DiffMethod.FORWARD — 1 f-eval/param instead of 2 (noisier but cheaper).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        n_samples = len(X)
        best_loss = np.inf
        no_improve = 0
        self.loss_history = []
        loss = np.nan  # sentinel before first eval

        epoch_iter = range(n_iter)
        pbar = None
        if progress_bar and _TQDM_AVAILABLE:
            pbar = _tqdm(epoch_iter, desc="Fitting", unit="epoch")
            epoch_iter = pbar

        for epoch in epoch_iter:
            current_lr = self.lr_schedule(lr, epoch)

            # ---- Mini-batch sampling ----
            if batch_size is not None and batch_size < n_samples:
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch, Y_batch = X[idx], Y[idx]
            else:
                X_batch, Y_batch = X, Y

            # ---- Gradient step ----
            grads = self._compute_gradients(X_batch, Y_batch)
            self._opt_step_fn(self.p, grads, self._opt_state, current_lr)

            # ---- Track loss (skip if not an eval epoch) ----
            if epoch % eval_every == 0:
                if validation_data is not None:
                    X_v, Y_v = validation_data
                    loss = self._objective(self.p, np.asarray(X_v, float), np.asarray(Y_v, float))
                else:
                    loss = self._objective(self.p, X, Y)
                self.loss_history.append(loss)

                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{current_lr:.2e}")

                if verbose and epoch % max(eval_every, 10) == 0:
                    print(f"Epoch {epoch:5d}  loss={loss:.6f}  lr={current_lr:.2e}")

                # ---- Early stopping ----
                if early_stopping:
                    if best_loss - loss > tol:
                        best_loss = loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (no improvement for {patience} steps)")
                        break

        if pbar is not None:
            pbar.close()

        return self

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_loss(self, X, Y):
        """Compute the current objective (data loss + regularization)."""
        return self._objective(self.p, np.asarray(X, float), np.asarray(Y, float))

    def get_params(self):
        """Return a copy of the current parameter dict."""
        return {k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self.p.items()}

    # ------------------------------------------------------------------
    # ONNX export / import
    # ------------------------------------------------------------------

    def to_onnx(self, path=None, *, input_shape, input_name="X", opset=17):
        """
        Export this model to ONNX format.

        Requires the model to have been created with ``vectorized=True`` and
        the function ``f(X, p)`` to use standard numpy operations (matmul,
        exp, tanh, sin, etc.) that the tracer can convert to ONNX nodes.

        Parameters
        ----------
        path        : str or None
            File path to write the ``.onnx`` file.  Returns the proto without
            writing if None.
        input_shape : tuple
            Shape of a *single* input sample, e.g. ``(2,)`` for 2 features.
            The batch dimension is added automatically (dynamic).
        input_name  : str
            Name of the ONNX input tensor (default ``"X"``).
        opset       : int
            ONNX opset version (default 17).

        Returns
        -------
        onnx.ModelProto

        Raises
        ------
        OnnxTraceError
            If the model is not vectorized or uses unsupported numpy ops.
        ImportError
            If ``onnx`` is not installed (``pip install lambdaml[onnx]``).

        Examples
        --------
        >>> proto = model.to_onnx('model.onnx', input_shape=(2,))

        See Also
        --------
        save_params : always-works alternative that saves weights as ``.npz``.
        """
        return _onnx_module.to_onnx(
            self, path,
            input_shape=input_shape,
            input_name=input_name,
            opset=opset,
        )

    def save_params(self, path, **metadata):
        """
        Save model parameters to a compressed ``.npz`` file.

        This always works — no ONNX tracing, no vectorized requirement.
        On load you must reconstruct the model (same ``f`` and initial ``p``
        structure) and call ``load_params()`` to restore weights.

        Parameters
        ----------
        path     : str — output file path (adds ``.npz`` if missing).
        **metadata : extra values to store (e.g. ``model_type='classifier'``).

        Examples
        --------
        >>> model.save_params('weights.npz')
        >>> model2.load_params('weights.npz')   # restore into a fresh instance
        """
        _onnx_module.save_params(self, path, **metadata)

    def load_params(self, path):
        """
        Load parameters from a ``.npz`` file saved by ``save_params()``.

        The model must have been constructed with the same parameter keys.
        Updates ``self.p`` in-place and returns ``self``.

        Parameters
        ----------
        path : str — path to the ``.npz`` file.

        Returns
        -------
        self

        Examples
        --------
        >>> model2 = LambdaClassifierModel(f=my_f, p=p_init, vectorized=True)
        >>> model2.load_params('weights.npz')
        >>> preds = model2.predict(X_test)
        """
        _onnx_module.load_params(self, path)
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────

class LambdaClassifierModel(_LambdaBaseModel):
    """
    A generic binary classifier.

    Your function f(x, p) must return a value in [0, 1] representing P(y=1|x).
    Objective: binary cross-entropy + regularization (minimized).

    Example
    -------
    >>> def logistic(x, p):
    ...     return 1 / (1 + np.exp(-p['w'].dot(x) - p['b']))
    >>> model = LambdaClassifierModel(f=logistic, p={'w': np.zeros(2), 'b': 0.0})
    >>> model.fit(X_train, Y_train, n_iter=200, lr=0.01, optimizer='adam')
    >>> preds = model.predict(X_test)
    """

    def __init__(self, f, p, **kwargs):
        # Back-compat: original had positional l1_factor, l2_factor
        # We allow them as keyword args through **kwargs
        super().__init__(f, p, **kwargs)

    def _loss_fn(self, y_pred, y_true):
        return LossFunctions.binary_cross_entropy(y_pred, y_true)

    def predict_proba(self, X, progress_bar=False):
        """
        Return P(y=1|x) for each sample.

        Parameters
        ----------
        progress_bar : bool — show a per-sample tqdm bar (default False)
        """
        X = np.asarray(X, dtype=float)
        return self._predict_batch(X, self.p,
                                   show_progress=progress_bar,
                                   progress_desc="Predicting proba")

    def predict(self, X, threshold=0.5, progress_bar=False):
        """
        Return binary class labels (0 or 1).

        Parameters
        ----------
        threshold    : float — decision boundary (default 0.5)
        progress_bar : bool  — show a per-sample tqdm bar (default False)
        """
        return (self.predict_proba(X, progress_bar=progress_bar) >= threshold).astype(int)

    def score(self, X, Y):
        """Classification accuracy."""
        return np.mean(self.predict(X) == np.asarray(Y))

    def compute_log_likelihood(self, X, Y):
        """
        Convenience wrapper: returns the (positive) log-likelihood on (X, Y).
        Kept for backward compatibility with original API.
        """
        y_pred = self._predict_batch(np.asarray(X, float), self.p)
        return LossFunctions.log_likelihood(y_pred, np.asarray(Y, float))

    # Backward-compatibility alias
    def predict_probability(self, X):
        """Alias for predict_proba (backward compat with original API)."""
        return list(self.predict_proba(X))


# ──────────────────────────────────────────────────────────────────────────────
# Regressor
# ──────────────────────────────────────────────────────────────────────────────

class LambdaRegressorModel(_LambdaBaseModel):
    """
    A generic regressor.

    Your function f(x, p) can return any real value.
    Objective: configurable loss (mse | mae | huber) + regularization (minimized).

    Example
    -------
    >>> def sine_reg(x, p):
    ...     return np.sin(p['w'].dot(x) + p['b'])
    >>> model = LambdaRegressorModel(f=sine_reg, p={'w': np.ones(2), 'b': 0.0},
    ...                              loss='mse')
    >>> model.fit(X_train, Y_train, n_iter=300, lr=1e-3)
    >>> preds = model.predict(X_test)
    """

    _LOSS_FNS = {
        "mse":         LossFunctions.mse,
        "mae":         LossFunctions.mae,
        "huber":       LossFunctions.huber,
        "pseudo_huber": LossFunctions.pseudo_huber,
    }

    def __init__(self, f, p, *, loss="mse", huber_delta=1.0, **kwargs):
        super().__init__(f, p, **kwargs)
        if loss not in self._LOSS_FNS:
            raise ValueError(f"loss must be one of {list(self._LOSS_FNS)}; got {loss!r}\n"
                             "Tip: use 'pseudo_huber' instead of 'huber' with DiffMethod.COMPLEX_STEP.")
        self._loss_name  = loss
        self._huber_delta = huber_delta

    def _loss_fn(self, y_pred, y_true):
        if self._loss_name == "huber":
            return LossFunctions.huber(y_pred, y_true, self._huber_delta)
        return self._LOSS_FNS[self._loss_name](y_pred, y_true)

    def predict(self, X, progress_bar=False):
        """
        Return continuous predictions.

        Parameters
        ----------
        progress_bar : bool — show a per-sample tqdm bar (default False)
        """
        X = np.asarray(X, dtype=float)
        return self._predict_batch(X, self.p,
                                   show_progress=progress_bar,
                                   progress_desc="Predicting")

    def score(self, X, Y):
        """R² score."""
        y_pred = self.predict(X)
        y_true = np.asarray(Y, float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
