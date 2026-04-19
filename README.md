# LambdaML

**Gradient-free machine learning. Give it any function; it learns the parameters.**

LambdaML lets you use *any numpy-compatible function* as your model and automatically fits its parameters using **numerical (finite-difference) differentiation** — no hand-derived gradients required. The "lambda" really can be anything: logistic regression, a neural network with custom activations, a physics equation, a learnable signal transform, or something entirely your own.

---

## Quick-start

```bash
pip install lambdaml           # core (numpy only)
pip install lambdaml[speed]    # + tqdm progress bars + joblib parallelism (recommended)
pip install lambdaml[onnx]     # + ONNX export/import (v1.2.0)
pip install lambdaml[examples] # + scipy, pandas, matplotlib for the notebook
pip install lambdaml[all]      # everything
```

```python
import numpy as np
from lambdaml import LambdaClassifierModel, Optimizer, DiffMethod, LRSchedule

# 1. Write your model — anything numpy-compatible works
def my_model(x, p):
    return (np.tanh(p['w'].dot(x) + p['b']) + 1) / 2

# 2. Initial parameters (scalars or numpy arrays)
p = {'w': np.zeros(2), 'b': 0.0}

# 3. Create and fit
model = LambdaClassifierModel(
    f=my_model,
    p=p,
    diff_method=DiffMethod.COMPLEX_STEP,   # recommended
    l2_factor=0.001,
    optimizer=Optimizer.ADAM,
    lr_schedule=LRSchedule.cosine_annealing(T_max=100),
    n_jobs=-1,   # parallel gradient computation across parameters (requires joblib)
)
model.fit(X_train, Y_train, n_iter=100, lr=0.01,
          early_stopping=True, patience=10)

print(model.score(X_test, Y_test))       # accuracy
print(model.predict_proba(X_test))       # probabilities
```

For regression, swap in `LambdaRegressorModel` with `loss='mse'`, `'mae'`, `'huber'`, or `'pseudo_huber'`.

See the [`examples/`](examples/) folder for runnable scripts and [`LambdaML_Showcase.ipynb`](LambdaML_Showcase.ipynb) for an interactive walkthrough with charts.

To export a fitted model to ONNX:

```python
# Requires vectorized=True on the model
proto = model.to_onnx('model.onnx', input_shape=(2,))

# Inference via onnxruntime (no Python/LambdaML needed at runtime)
from lambdaml import predict_onnx
probs = predict_onnx('model.onnx', X_test)

# Or save/load weights as .npz (always works, any model)
model.save_params('weights.npz')
model2.load_params('weights.npz')
```

---

## What's new in v1.2.0

**ONNX export and model persistence** — two new ways to save and deploy fitted models:

`to_onnx()` traces a fitted vectorized model to a standard ONNX graph with parameters baked in as initializers. The exported file runs anywhere `onnxruntime` is installed — no Python model code, no LambdaML, no numpy required at inference time.

`save_params()` / `load_params()` always works — saves the parameter dict to a compressed `.npz` file regardless of whether the model is vectorized. Requires re-providing the function `f` at load time (keeps the function in source, weights in the file).

```bash
pip install lambdaml[onnx]   # onnx + onnxruntime
```

```python
# Auto-trace export (vectorized models only)
def logistic_v(X, p):
    return 1 / (1 + np.exp(-(X @ p['w'] + p['b'])))

model = LambdaClassifierModel(f=logistic_v, p={'w': np.zeros(2), 'b': 0.0},
                              vectorized=True)
model.fit(X_train, Y_train, n_iter=200, lr=0.01)
model.to_onnx('logistic.onnx', input_shape=(2,))

# Runtime inference (no LambdaML needed)
import onnxruntime as rt
sess = rt.InferenceSession('logistic.onnx')
probs = sess.run(None, {'X': X_test.astype('float32')})[0]

# Convenience wrapper
from lambdaml import predict_onnx, from_onnx
probs  = predict_onnx('logistic.onnx', X_test)           # raw probabilities
labels = predict_onnx('logistic.onnx', X_test, threshold=0.5)  # binary labels

# Parameters-only (always works)
model.save_params('weights.npz')
model2 = LambdaClassifierModel(f=logistic_v, p={'w': np.zeros(2), 'b': 0.0},
                               vectorized=True)
model2.load_params('weights.npz')
```

The tracer supports standard numpy ops: `@` / `dot`, `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sqrt`, `tanh`, `sin`, `cos`, `abs`, `clip`, `log1p`, integer indexing (`X[:, 0]`), and slicing. Physics models using `arctan2` or complex control flow should use `save_params()` instead.

---

## What's new in v1.1.0

**Modify-and-restore gradient computation** — the single biggest internal speedup. Previously, every single parameter perturbation during gradient computation made a full deep copy of the entire parameter dictionary. Now the library perturbs one value in-place, evaluates, and restores — zero unnecessary copies. 2–5× speedup on the gradient step alone.

**Cached skip set & optimizer dispatch** — the regularization skip set (which parameters to exclude) and the optimizer update function are both resolved once at model construction time. Previously they were recomputed or re-dispatched via string comparison on every step of every epoch.

**Single-pass regularization** — when both L1 and L2 are active, the penalty is now computed in a single loop over parameters (was two separate passes before).

**`eval_every=10` by default** — each loss evaluation is a full forward pass over your entire dataset. Defaulting to every-10-epochs avoids 90% of those evaluations out of the box, with no effect on gradient quality.

**`n_jobs` — parallel gradient computation** — set `n_jobs=-1` to use all CPU cores. Each parameter's gradient is independent of every other, so this is embarrassingly parallel. Requires `joblib` (`pip install lambdaml[speed]`).

```python
# All v1.1.0 speed features together:
def f(X, p):                         # vectorized: accepts full matrix
    return X @ p['w'] + p['b']

model = LambdaRegressorModel(
    f=f,
    p={'w': np.zeros(10), 'b': 0.0},
    vectorized=True,   # eliminates Python sample loop
    n_jobs=-1,         # parallelise gradient across parameters
)
model.fit(X, Y, n_iter=200, lr=0.01, eval_every=50)
```

**v1.0.3 additions** (still available): progress bars (`progress_bar=True` on `fit`/`predict`), `eval_every` parameter, `vectorized=True` mode.

---

## What is finite-difference differentiation?

The term you're looking for is **finite-difference approximation** (sometimes called *numerical differentiation*). Rather than deriving f′(θ) analytically, we estimate it by evaluating the function at nearby points:

```
f'(θ) ≈ [f(θ+h) - f(θ-h)] / (2h)     ← Central difference, O(h²)
```

LambdaML supports six methods with different accuracy/cost trade-offs:

| Method | Order | f-evals/param | Notes |
|---|---|---|---|
| Forward | O(h) | 1 | Fast, low accuracy |
| Backward | O(h) | 1 | Fast, low accuracy |
| Central | O(h²) | 2 | Default — good balance |
| Five-Point | O(h⁴) | 4 | High accuracy, smooth f |
| **Complex-Step** | O(h²) | 1 (complex) | **Recommended** — no cancellation error |
| Richardson | O(h⁴) | 4 | High accuracy, no complex inputs needed |

**Is it tractable?** Yes, for models up to ~10k parameters. Each gradient step costs O(n_params) forward passes instead of O(1) for analytic backprop. For small-to-medium models on a CPU+numpy backend this is entirely practical.

---

## Speed tips

The main cost per epoch is `n_params × diff_evals × n_samples` calls to `f`. To reduce it:

| Technique | How | Typical gain |
|---|---|---|
| Modify-and-restore *(v1.1.0, automatic)* | No dict copies during gradient computation | 2–5× |
| `n_jobs=-1` *(v1.1.0)* | Parallel gradient across parameters via joblib | ~N_cores× |
| `vectorized=True` *(v1.0.3)* | Write `f(X, p)` to accept the full matrix | 2–10× |
| `eval_every=50` | Skip loss re-evaluation on most epochs | ~1.5–2× |
| `batch_size=N` | Mini-batch gradient steps | Scales with batch ratio |
| `DiffMethod.FORWARD` | 1 f-eval/param instead of 2 | ~1.5× (noisier grads) |

---

## The lambda can be any function

Six completely different model functions, one `.fit()` call — logistic regression, tanh, sine activation (non-standard), Gaussian RBF, softplus, and a physics-inspired decay+oscillation model `σ(a·exp(−λ|x₀|)·cos(ω·x₁+φ))`. See `LambdaML_Showcase.ipynb` for visualisations and benchmarks.

---

## API reference

### `LambdaClassifierModel(f, p, **kwargs)`

| Parameter | Default | Description |
|---|---|---|
| `f` | — | Model: `f(x, p) → float ∈ (0,1)` — or `f(X, p) → array` when `vectorized=True` |
| `p` | — | Parameter dict (scalars or numpy arrays) |
| `diff_method` | `DiffMethod.CENTRAL` | Finite-difference method |
| `diff_h` | `None` | Custom step size (None = optimal default per method) |
| `l1_factor` | `0.0` | L1 regularization strength |
| `l2_factor` | `0.01` | L2 regularization strength |
| `regularize_bias` | `False` | Whether to regularize `b*` params |
| `optimizer` | `Optimizer.ADAM` | `sgd`, `momentum`, `rmsprop`, `adam` |
| `lr_schedule` | `None` (constant) | Learning rate schedule callable |
| `vectorized` | `False` | If `True`, `f` receives the full `X` matrix — faster |
| `n_jobs` | `1` | Parallel gradient workers; `-1` = all cores (requires `joblib`) |

**`.fit(X, Y, ...)`**

| Parameter | Default | Description |
|---|---|---|
| `n_iter` | `100` | Max gradient steps |
| `lr` | `0.01` | Initial learning rate |
| `batch_size` | `None` | Mini-batch size; `None` = full batch |
| `early_stopping` | `False` | Stop if loss stalls for `patience` steps |
| `patience` | `10` | Early stopping patience |
| `tol` | `1e-6` | Minimum improvement threshold |
| `verbose` | `False` | Print loss every `eval_every` iterations |
| `validation_data` | `None` | `(X_val, Y_val)` tuple |
| `progress_bar` | `True` | Show tqdm epoch bar (requires `tqdm`) |
| `eval_every` | `10` | Evaluate loss every N epochs — each eval is a full forward pass |

**Other methods:** `.predict(X, progress_bar=False)` · `.predict_proba(X, progress_bar=False)` · `.score(X, Y)` · `.compute_loss(X, Y)` · `.get_params()` · `.loss_history`

**ONNX / persistence methods** (on both model classes):

| Method | Description |
|---|---|
| `.to_onnx(path, *, input_shape, ...)` | Export to ONNX. Requires `vectorized=True`. Returns `onnx.ModelProto`. |
| `.save_params(path, **meta)` | Save weights to `.npz`. Always works. |
| `.load_params(path)` | Load weights from `.npz` into this model. Returns `self`. |

**Module-level helpers** (importable from `lambdaml`):

| Function | Description |
|---|---|
| `predict_onnx(path_or_session, X, threshold=None)` | Run ONNX inference. Returns `(n,)` array. |
| `from_onnx(path)` | Load an ONNX file and return an `onnxruntime.InferenceSession`. |
| `save_params(model, path, **meta)` | Functional form of `.save_params()`. |
| `load_params(model, path)` | Functional form of `.load_params()`. |
| `OnnxTraceError` | Exception raised when tracing fails. |

### `LambdaRegressorModel(f, p, loss='mse', **kwargs)`

| Parameter | Default | Description |
|---|---|---|
| `loss` | `'mse'` | `'mse'`, `'mae'`, `'huber'`, `'pseudo_huber'` |
| `huber_delta` | `1.0` | Threshold for Huber / pseudo-Huber |

**Methods:** `.fit(...)` · `.predict(X, progress_bar=False)` · `.score(X, Y)` (R²)

### `DiffMethod` · `Optimizer` · `LRSchedule`

```python
# Derivative methods
DiffMethod.FORWARD | BACKWARD | CENTRAL | FIVE_POINT | COMPLEX_STEP | RICHARDSON

# Optimizers
Optimizer.SGD | MOMENTUM | RMSPROP | ADAM

# LR schedules
LRSchedule.constant()
LRSchedule.step_decay(drop=0.5, epochs_drop=10)
LRSchedule.exponential_decay(k=0.01)
LRSchedule.cosine_annealing(T_max=100)
LRSchedule.warmup_cosine(warmup=10, T_max=100)
```

---

## Performance improvements over original

| Change | Original | v1.1.0 |
|---|---|---|
| Gradient dict copies | Full dict copy per parameter element | Modify-and-restore — zero copies |
| Regularization passes | 1–2 passes per epoch (separate L1/L2 loops) | Single pass regardless of combination |
| Optimizer dispatch | String compare per parameter per step | Function resolved once at init |
| Skip set computation | Rebuilt from scratch every epoch | Cached frozenset at init |
| Default loss eval frequency | Every epoch (`eval_every=1`) | Every 10 epochs (`eval_every=10`) |
| Gradient parallelism | Sequential across all parameters | Optional: `n_jobs=-1` uses all CPU cores |

## ONNX support (v1.2.0)

| Feature | Details |
|---|---|
| Export method | `model.to_onnx(path, input_shape=(n_features,))` |
| Requirement | `vectorized=True` on the model; function uses standard numpy ops |
| Serialization | Parameters baked in as ONNX float32 initializers |
| ONNX opset | 17 (default); IR version 8 for broad onnxruntime compatibility |
| Inference | `onnxruntime` — no LambdaML, no numpy, no Python model code needed |
| Fallback | `save_params()` / `load_params()` — always works, any model type |
| Install | `pip install lambdaml[onnx]` |
| Supported numpy ops | `@`, `dot`, `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sqrt`, `tanh`, `sin`, `cos`, `abs`, `clip`, `log1p`, integer indexing `X[:, i]`, slicing |

---

## Bug fixes from the original library

| Bug | Original | Fixed |
|---|---|---|
| Epsilon | `float16.eps ≈ 0.001` — catastrophically large | Float64-optimal per method (~6e-6 for central) |
| L1 regularization | Summed raw `θ` — negative weights reduced penalty | Summed `\|θ\|` using smooth complex-safe approximation |
| Closure-in-loop | Array gradient loop captured last index for all closures | Fixed with factory functions |
| L1/L2 complex-step safety | `float()` cast stripped imaginary part | Uses `v*v` and `sqrt(v*v+eps)` to preserve imaginary parts |
| No test split | Accuracy reported on training data | Train/test split in all examples |

---

## Is LambdaML useful for Kaggling?

**As a primary model for large nets — rarely. As a prototyping and ensembling tool — genuinely yes.**

The core insight: LambdaML *decouples your model definition from gradient computation*. Anywhere you want a custom functional form but don't want to derive its gradients by hand, LambdaML fills that gap.

Concrete use cases: fitting domain equations with unknown parameters (physics-based pricing, pharmacokinetics, decay curves); directly optimising non-differentiable competition metrics (NDCG, F-beta, Cohen's kappa) as the loss function; building exotic meta-learners in stacking ensembles; small-data + custom hypothesis problems where sklearn doesn't have your model form.

---

## Project structure

```
LambdaML/
├── lambdaml/                # Installable package (pip install lambdaml)
│   ├── __init__.py
│   ├── lambda_model.py      # LambdaClassifierModel, LambdaRegressorModel, Optimizer
│   ├── lambda_utils.py      # NumericalDiff, GradientComputer, Regularization, LossFunctions, LRSchedule
│   └── lambda_onnx.py       # to_onnx, from_onnx, save_params, load_params, predict_onnx (v1.2.0)
├── pyproject.toml           # Package metadata
├── LambdaML_Showcase.ipynb  # Interactive notebook with all charts (section 12: ONNX)
├── examples/
│   ├── example_tanh_regression.py
│   ├── example_neural_network.py
│   ├── example_diff_methods.py
│   ├── example_regressor.py
│   └── example_onnx.py      # ONNX export/import demo and benchmark (v1.2.0)
├── data/
│   └── circles.csv
└── legacy/                  # Original library files (pre-rewrite)
```

---

## License

See [`LICENSE`](LICENSE).
