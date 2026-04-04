"""
example_tanh_regression.py
--------------------------
Demonstrates LambdaML on a linearly-separable binary classification task using
a custom tanh-based decision boundary — a function that would require manual
gradient derivation in any analytical framework.

With LambdaML you just write f(x, p) and call .fit().
"""

import numpy as np
from lambdaml import LambdaClassifierModel, Optimizer, DiffMethod, LRSchedule


def synthesize_data(n=2000, seed=42):
    """Two linearly-separable Gaussian clusters."""
    np.random.seed(seed)
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], n)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], n)
    X = np.vstack((x1, x2)).astype(np.float64)
    Y = np.hstack((np.zeros(n), np.ones(n)))
    return X, Y


def tanh_regression(x, p):
    """Custom tanh-based model: output ∈ (0, 1)."""
    signal = p['w'].dot(x) + p['b']
    return (np.tanh(signal) + 1) / 2


# ── Data ──────────────────────────────────────────────────────────────────────
X, Y = synthesize_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# ── Parameters ────────────────────────────────────────────────────────────────
np.random.seed(0)
p = {
    'w': np.random.randn(X.shape[1]) * 0.01,
    'b': 0.0,
}

# ── Model ─────────────────────────────────────────────────────────────────────
model = LambdaClassifierModel(
    f=tanh_regression,
    p=p,
    diff_method=DiffMethod.COMPLEX_STEP,   # most accurate; no cancellation error
    l2_factor=0.001,                        # light Ridge regularization
    optimizer=Optimizer.ADAM,
    lr_schedule=LRSchedule.cosine_annealing(T_max=50),
)

# ── Fit ───────────────────────────────────────────────────────────────────────
print("=== Tanh Regression Example ===")
print(f"Initial log-likelihood : {model.compute_log_likelihood(X_train, Y_train):.4f}")

model.fit(
    X_train, Y_train,
    n_iter=50,
    lr=0.05,
    early_stopping=True,
    patience=8,
    verbose=True,
    validation_data=(X_test, Y_test),
)

print(f"Final   log-likelihood : {model.compute_log_likelihood(X_train, Y_train):.4f}")
print(f"Train accuracy         : {model.score(X_train, Y_train):.4f}")
print(f"Test  accuracy         : {model.score(X_test,  Y_test):.4f}")
print(f"Learned weights        : {model.p['w']}")
print(f"Learned bias           : {model.p['b']:.4f}")
