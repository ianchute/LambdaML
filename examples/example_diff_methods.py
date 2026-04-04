"""
example_diff_methods.py
-----------------------
Compares all six finite-difference methods on the same logistic regression task.
Useful for understanding the accuracy/speed trade-off of each gradient estimator.

Methods compared
----------------
  FORWARD      — O(h), 1 f-eval per param
  BACKWARD     — O(h), 1 f-eval per param
  CENTRAL      — O(h²), 2 f-evals per param
  FIVE_POINT   — O(h⁴), 4 f-evals per param
  COMPLEX_STEP — O(h²) with zero cancellation error, 1 f-eval per param (complex)
  RICHARDSON   — O(h⁴) via Richardson extrapolation, 4 f-evals per param
"""

import numpy as np
import time
from lambdaml import LambdaClassifierModel, Optimizer, DiffMethod



def synthesize_data(n=500, seed=7):
    np.random.seed(seed)
    x1 = np.random.multivariate_normal([0, 0], [[1, .5], [.5, 1]], n)
    x2 = np.random.multivariate_normal([2, 2], [[1, .5], [.5, 1]], n)
    X = np.vstack((x1, x2))
    Y = np.hstack((np.zeros(n), np.ones(n)))
    return X, Y


def logistic(x, p):
    return 1.0 / (1.0 + np.exp(-p['w'].dot(x) - p['b']))


X, Y = synthesize_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

methods = [
    DiffMethod.FORWARD,
    DiffMethod.BACKWARD,
    DiffMethod.CENTRAL,
    DiffMethod.FIVE_POINT,
    DiffMethod.COMPLEX_STEP,
    DiffMethod.RICHARDSON,
]

print(f"{'Method':<20} {'Train Acc':>10} {'Test Acc':>10} {'Final Loss':>12} {'Time (s)':>10}")
print("-" * 66)

for method in methods:
    np.random.seed(42)
    p = {'w': np.zeros(X.shape[1]), 'b': 0.0}

    model = LambdaClassifierModel(
        f=logistic, p=p,
        diff_method=method,
        l2_factor=0.001,
        optimizer=Optimizer.ADAM,
    )

    t0 = time.perf_counter()
    model.fit(X_train, Y_train, n_iter=50, lr=0.05, verbose=False)
    elapsed = time.perf_counter() - t0

    train_acc = model.score(X_train, Y_train)
    test_acc  = model.score(X_test, Y_test)
    loss      = model.compute_loss(X_test, Y_test)

    print(f"{method:<20} {train_acc:>10.4f} {test_acc:>10.4f} {loss:>12.6f} {elapsed:>10.3f}")
