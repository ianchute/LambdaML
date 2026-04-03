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
"""

import numpy as np
from lambda_utils import (
    GradientComputer, DiffMethod,
    Regularization, LossFunctions, LRSchedule
)


# ──────────────────────────────────────────────────────────────────────────────
# Optimizers
# ──────────────────────────────────────────────────────────────────────────────

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

        self._gc = GradientComputer(method=diff_method, h=diff_h)
        self._opt_state = _OptimizerState(self.p)

        # History
        self.loss_history = []

    # ------------------------------------------------------------------
    # Regularization penalty (added to loss, not subtracted from log-lik)
    # ------------------------------------------------------------------

    def _reg_skip(self):
        if self.regularize_bias:
            return set()
        # Skip parameters whose key starts with 'b' (bias convention)
        return {k for k in self.p if k.startswith('b')}

    def _reg_penalty(self, p):
        skip = self._reg_skip()
        penalty = 0.0
        if self.l1_factor > 0 and self.l2_factor > 0:
            penalty = (self.l1_factor * Regularization.l1(p, skip) +
                       self.l2_factor * Regularization.l2(p, skip))
        elif self.l1_factor > 0:
            penalty = self.l1_factor * Regularization.l1(p, skip)
        elif self.l2_factor > 0:
            penalty = self.l2_factor * Regularization.l2(p, skip)
        return penalty

    # ------------------------------------------------------------------
    # Prediction (subclasses implement _predict_batch)
    # ------------------------------------------------------------------

    def _predict_batch(self, X, p):
        """Return array of predictions for all rows in X."""
        return np.array([self.f(x, p) for x in X])

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
        grads = {}
        for key in self.p:
            grads[key] = self._gc.compute(self._objective, self.p, key, X, Y)
        return grads

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
        verbose      : bool — print loss every 10 iterations
        validation_data : tuple (X_val, Y_val) or None

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        n_samples = len(X)
        best_loss = np.inf
        no_improve = 0
        self.loss_history = []

        for epoch in range(n_iter):
            current_lr = self.lr_schedule(lr, epoch)

            # ---- Mini-batch sampling ----
            if batch_size is not None and batch_size < n_samples:
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch, Y_batch = X[idx], Y[idx]
            else:
                X_batch, Y_batch = X, Y

            # ---- Gradient step ----
            grads = self._compute_gradients(X_batch, Y_batch)
            Optimizer.step(
                self.optimizer_name, self.p, grads, self._opt_state, current_lr,
                momentum=self.momentum, beta1=self.beta1, beta2=self.beta2,
                epsilon=self.adam_eps,
            )

            # ---- Track loss (always on full data for consistency) ----
            if validation_data is not None:
                X_v, Y_v = validation_data
                loss = self._objective(self.p, np.asarray(X_v, float), np.asarray(Y_v, float))
            else:
                loss = self._objective(self.p, X, Y)

            self.loss_history.append(loss)

            if verbose and epoch % 10 == 0:
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

    def predict_proba(self, X):
        """Return P(y=1|x) for each sample."""
        X = np.asarray(X, dtype=float)
        return self._predict_batch(X, self.p)

    def predict(self, X, threshold=0.5):
        """Return binary class labels (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

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

    def predict(self, X):
        """Return continuous predictions."""
        X = np.asarray(X, dtype=float)
        return self._predict_batch(X, self.p)

    def score(self, X, Y):
        """R² score."""
        y_pred = self.predict(X)
        y_true = np.asarray(Y, float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
