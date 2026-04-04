"""
example_neural_network.py
--------------------------
Demonstrates LambdaML fitting a small 2-layer neural network on the non-linearly
separable "circles" dataset — entirely without writing any analytical gradients.

The network uses ELU activations (non-standard) to show that LambdaML doesn't
care what function you use: any numpy-compatible black-box works.

Key differences from the legacy version
----------------------------------------
• Adam optimizer instead of vanilla SGD → faster convergence
• Central-difference gradient (with correct float64 epsilon — was float16!)
• Corrected L1 regularization (now uses |θ|, not θ)
• Mini-batch training for noise-regularized learning
• Early stopping to prevent overfitting
• Train/test split + accuracy reported on held-out data
"""

import numpy as np
import pandas as pd
from lambdaml import LambdaClassifierModel, Optimizer, DiffMethod, LRSchedule


def get_data():
    circles = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'circles.csv'))
    return circles[['x', 'y']].values.astype(float), circles['label'].values.astype(float)


# ── Architecture ──────────────────────────────────────────────────────────────

def elu(z, alpha=0.01):
    """Exponential Linear Unit — smooth alternative to ReLU."""
    return z if z >= 0 else alpha * (np.exp(z) - 1)


def neuron(x, p, w_key, b_key):
    return elu(p[w_key].dot(x) + p[b_key])


def hidden_layer(x, p):
    return np.array([
        neuron(x, p, 'w1', 'b1'),
        neuron(x, p, 'w2', 'b2'),
        neuron(x, p, 'w3', 'b3'),   # extra hidden unit vs legacy
    ])


def neural_network(x, p):
    """2-layer ELU network → sigmoid output."""
    h = hidden_layer(x, p)
    signal = p['wf'].dot(h) + p['bf']
    return (np.tanh(signal) + 1) / 2


# ── Data ──────────────────────────────────────────────────────────────────────
np.random.seed(1)
X, Y = get_data()
n = len(X)
idx = np.random.permutation(n)
split = int(0.8 * n)
X_train, X_test = X[idx[:split]], X[idx[split:]]
Y_train, Y_test = Y[idx[:split]], Y[idx[split:]]

n_features = X.shape[1]

# ── Parameters ────────────────────────────────────────────────────────────────
def rand_w(size):
    return np.random.randn(size) * np.sqrt(2.0 / size)   # He init

p = {
    'w1': rand_w(n_features), 'b1': 0.0,
    'w2': rand_w(n_features), 'b2': 0.0,
    'w3': rand_w(n_features), 'b3': 0.0,
    'wf': rand_w(3),           'bf': 0.0,
}

# ── Model ─────────────────────────────────────────────────────────────────────
model = LambdaClassifierModel(
    f=neural_network,
    p=p,
    diff_method=DiffMethod.CENTRAL,    # O(h²), no complex number required
    l2_factor=0.005,
    l1_factor=0.0,
    optimizer=Optimizer.ADAM,
    lr_schedule=LRSchedule.step_decay(drop=0.5, epochs_drop=50),
)

# ── Fit ───────────────────────────────────────────────────────────────────────
print("=== Neural Network on Circles Dataset ===")
print(f"Initial log-likelihood : {model.compute_log_likelihood(X_train, Y_train):.4f}")

model.fit(
    X_train, Y_train,
    n_iter=200,
    lr=0.01,
    batch_size=64,
    early_stopping=True,
    patience=20,
    tol=1e-5,
    verbose=True,
    validation_data=(X_test, Y_test),
)

print(f"Final log-likelihood   : {model.compute_log_likelihood(X_train, Y_train):.4f}")
print(f"Train accuracy         : {model.score(X_train, Y_train):.4f}")
print(f"Test  accuracy         : {model.score(X_test,  Y_test):.4f}")
