"""
example_regressor.py
--------------------
Demonstrates LambdaRegressorModel on a noisy sine-wave regression task.

Highlights:
  • Custom sine-based model function (not available in sklearn)
  • Huber loss for robustness to outliers
  • Adam optimizer + cosine annealing
  • R² score on held-out test set
"""

import numpy as np
from lambdaml import LambdaRegressorModel, Optimizer, DiffMethod, LRSchedule



# ── Data: noisy sine wave ──────────────────────────────────────────────────────
np.random.seed(0)
n = 400
X_raw = np.linspace(-3, 3, n)
Y_raw = np.sin(X_raw) + np.random.normal(0, 0.2, n)

# Add some outliers to showcase Huber robustness
outlier_idx = np.random.choice(n, 20, replace=False)
Y_raw[outlier_idx] += np.random.choice([-4, 4], 20)

# Reshape to (n, 1)
X = X_raw[:, None]
Y = Y_raw

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]


# ── Custom model: learnable frequency + phase ──────────────────────────────────
def sine_model(x, p):
    """f(x) = a·sin(ω·x + φ) + c"""
    return p['a'] * np.sin(p['omega'] * x[0] + p['phi']) + p['c']


p = {'a': 0.5, 'omega': 2.0, 'phi': 1.0, 'c': 0.5}   # deliberately off from truth

model = LambdaRegressorModel(
    f=sine_model,
    p=p,
    loss='pseudo_huber',     # smooth Huber — complex-step compatible (no np.where)
    huber_delta=1.0,
    diff_method=DiffMethod.COMPLEX_STEP,
    l2_factor=0.001,
    optimizer=Optimizer.ADAM,
    lr_schedule=LRSchedule.cosine_annealing(T_max=200),
)

print("=== Sine Regression with Huber Loss ===")
print(f"Initial loss : {model.compute_loss(X_train, Y_train):.4f}")

model.fit(
    X_train, Y_train,
    n_iter=200,
    lr=0.05,
    early_stopping=True,
    patience=20,
    verbose=True,
)

print(f"Final loss   : {model.compute_loss(X_train, Y_train):.4f}")
print(f"Train R²     : {model.score(X_train, Y_train):.4f}")
print(f"Test  R²     : {model.score(X_test,  Y_test):.4f}")
print(f"Learned params: a={model.p['a']:.3f}, ω={model.p['omega']:.3f}, "
      f"φ={model.p['phi']:.3f}, c={model.p['c']:.3f}")
print(f"True params:   a=1.000, ω=1.000, φ=0.000, c=0.000")
