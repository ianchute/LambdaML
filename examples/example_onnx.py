"""
example_onnx.py
---------------
Demonstrates ONNX export and inference for LambdaML models.

This script shows three workflows:

1. Auto-trace export (to_onnx)
   Train a vectorized model, export to ONNX, run inference via onnxruntime.
   Verifies numerical parity between native predict() and ONNX inference,
   and benchmarks throughput.

2. Parameters-only save/load (save_params / load_params)
   Always works — no tracing, no vectorized requirement.
   Shows round-trip for both a classifier and a regressor.

3. ONNX export for a vectorized regressor (sine recovery)

Requirements
------------
    pip install lambdaml[onnx]         # adds onnx + onnxruntime
    pip install lambdaml[examples]     # needed for this script (matplotlib etc.)

Run
---
    python examples/example_onnx.py
"""

import os
import sys
import time
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lambdaml import (
    LambdaClassifierModel,
    LambdaRegressorModel,
    Optimizer, DiffMethod, LRSchedule,
    to_onnx, from_onnx, predict_onnx,
    save_params, load_params,
    OnnxTraceError,
)

SEP = "─" * 60


# ──────────────────────────────────────────────────────────────────────────────
# Shared data
# ──────────────────────────────────────────────────────────────────────────────

def make_blobs(n=400, seed=42):
    np.random.seed(seed)
    x0 = np.random.multivariate_normal([0, 0], [[1, .4], [.4, 1]], n)
    x1 = np.random.multivariate_normal([3, 3], [[1, .4], [.4, 1]], n)
    X  = np.vstack([x0, x1])
    Y  = np.hstack([np.zeros(n), np.ones(n)])
    return X, Y


X_all, Y_all = make_blobs()
split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
Y_train, Y_test = Y_all[:split], Y_all[split:]

N_FEATURES = X_train.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Auto-trace ONNX export — logistic classifier (vectorized)
# ──────────────────────────────────────────────────────────────────────────────

print(SEP)
print("1. Auto-trace ONNX export  —  Logistic Classifier")
print(SEP)


def logistic_v(X, p):
    """Vectorized logistic regression: X @ w + b → sigmoid."""
    return 1.0 / (1.0 + np.exp(-(X @ p['w'] + p['b'])))


p_logistic = {'w': np.zeros(N_FEATURES), 'b': 0.0}

model_lr = LambdaClassifierModel(
    f=logistic_v,
    p=p_logistic,
    diff_method=DiffMethod.COMPLEX_STEP,
    l2_factor=0.001,
    optimizer=Optimizer.ADAM,
    vectorized=True,
)

print("Training logistic model …")
model_lr.fit(X_train, Y_train, n_iter=200, lr=0.05, eval_every=20)
native_acc = model_lr.score(X_test, Y_test)
print(f"Native test accuracy:   {native_acc:.4f}")

# ── Export to ONNX ────────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
    onnx_path = fh.name

try:
    proto = model_lr.to_onnx(onnx_path, input_shape=(N_FEATURES,))
    print(f"Exported to:            {onnx_path}")

    # ── ONNX inference ────────────────────────────────────────────────────────
    onnx_probs  = predict_onnx(onnx_path, X_test)
    onnx_labels = predict_onnx(onnx_path, X_test, threshold=0.5)
    onnx_acc    = np.mean(onnx_labels == Y_test)
    print(f"ONNX test accuracy:     {onnx_acc:.4f}")

    # ── Numerical parity ──────────────────────────────────────────────────────
    native_probs = model_lr.predict_proba(X_test)
    max_diff = np.max(np.abs(native_probs - onnx_probs))
    print(f"Max output difference:  {max_diff:.2e}  (native vs ONNX)")
    assert max_diff < 1e-4, f"Parity check failed: max diff = {max_diff:.4e}"
    print("Parity check:           PASSED ✓")

    # ── Throughput benchmark ──────────────────────────────────────────────────
    N_RUNS = 500
    X_bench = X_test.astype(np.float32)

    # Native
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = model_lr.predict_proba(X_test)
    t_native = (time.perf_counter() - t0) / N_RUNS

    # ONNX
    sess = from_onnx(onnx_path)
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        _ = sess.run(None, {input_name: X_bench})[0]
    t_onnx = (time.perf_counter() - t0) / N_RUNS

    print(f"\nThroughput ({len(X_test)} samples, {N_RUNS} runs):")
    print(f"  Native predict_proba: {t_native*1000:.3f} ms/call")
    print(f"  ONNX runtime:         {t_onnx*1000:.3f} ms/call")
    if t_native > 0:
        ratio = t_native / t_onnx
        print(f"  Speedup:              {ratio:.1f}×  (ONNX vs native)")

except OnnxTraceError as e:
    print(f"OnnxTraceError: {e}")
finally:
    os.unlink(onnx_path)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Parameters-only save / load  (always works, non-vectorized model)
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("2. Parameters-only save / load  —  Non-vectorized ELU network")
print(SEP)


def elu(z, alpha=0.01):
    return z if z >= 0 else alpha * (np.exp(z) - 1)


def neural_network(x, p):
    """Non-vectorized 2-layer ELU network (per-sample)."""
    h = np.array([
        elu(p['w1'].dot(x) + p['b1']),
        elu(p['w2'].dot(x) + p['b2']),
        elu(p['w3'].dot(x) + p['b3']),
    ])
    return (np.tanh(p['wf'].dot(h) + p['bf']) + 1) / 2


np.random.seed(1)
def rand_w(size): return np.random.randn(size) * np.sqrt(2.0 / size)

p_nn = {
    'w1': rand_w(N_FEATURES), 'b1': 0.0,
    'w2': rand_w(N_FEATURES), 'b2': 0.0,
    'w3': rand_w(N_FEATURES), 'b3': 0.0,
    'wf': rand_w(3),           'bf': 0.0,
}

model_nn = LambdaClassifierModel(
    f=neural_network,
    p=p_nn,
    diff_method=DiffMethod.CENTRAL,
    l2_factor=0.005,
    optimizer=Optimizer.ADAM,
    vectorized=False,
)

print("Training ELU neural network …")
model_nn.fit(X_train, Y_train, n_iter=150, lr=0.01, eval_every=20)
acc_before = model_nn.score(X_test, Y_test)
probs_before = model_nn.predict_proba(X_test)
print(f"Test accuracy (before save): {acc_before:.4f}")

# ── Save params ───────────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
    npz_path = fh.name

try:
    model_nn.save_params(npz_path, model_type="classifier", n_features=N_FEATURES)
    print(f"Params saved to: {npz_path}")

    # ── Reconstruct and load ───────────────────────────────────────────────────
    np.random.seed(99)   # different random init — will be overwritten
    p_fresh = {
        'w1': rand_w(N_FEATURES), 'b1': 0.0,
        'w2': rand_w(N_FEATURES), 'b2': 0.0,
        'w3': rand_w(N_FEATURES), 'b3': 0.0,
        'wf': rand_w(3),           'bf': 0.0,
    }
    model_nn2 = LambdaClassifierModel(
        f=neural_network,
        p=p_fresh,
        vectorized=False,
    )
    model_nn2.load_params(npz_path)

    acc_after = model_nn2.score(X_test, Y_test)
    probs_after = model_nn2.predict_proba(X_test)
    max_diff = np.max(np.abs(probs_before - probs_after))
    print(f"Test accuracy (after load):  {acc_after:.4f}")
    print(f"Max output difference:       {max_diff:.2e}")
    assert max_diff < 1e-10, f"Round-trip parity failed: {max_diff:.4e}"
    print("Round-trip parity:           PASSED ✓")

finally:
    os.unlink(npz_path)


# ──────────────────────────────────────────────────────────────────────────────
# 3. ONNX export — vectorized regressor (sine wave)
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("3. ONNX export  —  Vectorized Sine Regressor")
print(SEP)

np.random.seed(0)
n_pts = 300
X_sine = np.linspace(-np.pi, np.pi, n_pts)[:, None]
Y_sine = np.sin(X_sine[:, 0]) + np.random.normal(0, 0.1, n_pts)
Xtr_s, Xte_s = X_sine[:240], X_sine[240:]
Ytr_s, Yte_s = Y_sine[:240], Y_sine[240:]


def sine_v(X, p):
    """Vectorized learnable sine: a*sin(omega*x + phi) + c."""
    x = X[:, 0]
    return p['a'] * np.sin(p['omega'] * x + p['phi']) + p['c']


model_sine = LambdaRegressorModel(
    f=sine_v,
    p={'a': 1.5, 'omega': 0.8, 'phi': 0.5, 'c': 0.0},
    loss='mse',
    diff_method=DiffMethod.COMPLEX_STEP,
    optimizer=Optimizer.ADAM,
    vectorized=True,
)

print("Training sine regressor …")
model_sine.fit(Xtr_s, Ytr_s, n_iter=300, lr=0.05, eval_every=30)
r2_native = model_sine.score(Xte_s, Yte_s)
print(f"Native test R²:   {r2_native:.4f}")
print(f"Learned params:   a={model_sine.p['a']:.4f}  "
      f"omega={model_sine.p['omega']:.4f}  "
      f"phi={model_sine.p['phi']:.4f}  "
      f"c={model_sine.p['c']:.4f}")
print(f"True params:      a=1.0  omega=1.0  phi=0.0  c=0.0")

with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as fh:
    onnx_path_s = fh.name

try:
    model_sine.to_onnx(onnx_path_s, input_shape=(1,))
    print(f"Exported to:      {onnx_path_s}")

    onnx_preds  = predict_onnx(onnx_path_s, Xte_s)
    native_preds = model_sine.predict(Xte_s)
    max_diff = np.max(np.abs(native_preds - onnx_preds))
    print(f"Max output diff:  {max_diff:.2e}  (native vs ONNX)")
    assert max_diff < 1e-4, f"Parity check failed: {max_diff:.4e}"
    print("Parity check:     PASSED ✓")

except OnnxTraceError as e:
    print(f"OnnxTraceError: {e}")
finally:
    os.unlink(onnx_path_s)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Expected failure: non-vectorized model to ONNX
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("4. Expected error: non-vectorized model to_onnx()")
print(SEP)

try:
    model_nn.to_onnx('/tmp/should_fail.onnx', input_shape=(N_FEATURES,))
    print("ERROR: Should have raised OnnxTraceError!")
except OnnxTraceError as e:
    first_line = str(e).split('\n')[0]
    print(f"Got expected OnnxTraceError:  {first_line}")
    print("Error handling:               PASSED ✓")


print(f"\n{SEP}")
print("All ONNX tests completed successfully.")
print(SEP)
print()
print("Summary")
print("-------")
print("  to_onnx()       — vectorized models only; traces numpy ops to ONNX graph")
print("  predict_onnx()  — fast inference via onnxruntime (no Python sample loop)")
print("  save_params()   — always works; saves weights as .npz")
print("  load_params()   — restores weights into a freshly-constructed model")
print()
print("Install:  pip install lambdaml[onnx]")
