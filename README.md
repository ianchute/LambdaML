# LambdaML
### By: Ian Chu Te

<img src="https://maxcdn.icons8.com/windows10/PNG/512/Alphabet/lambda-512.png" width="128"/>

##### A lambda-based machine learning library.

> In computer programming, a *lambda function* is a function definition that is not bound to an identifier.

In LambdaML, you can use *any function (named or unnamed)* as your model function and LambdaML will automatically fit the parameters for you!

With LambdaML, all you need are two things:

1. **f** - your statistical model - the **"lambda"** (function):

```python

def sine_regression(x,p):
    """Custom sine-based regression model."""
    signal = p['w'].dot(x) + p['b']
    return sine_activation(signal)

```

2. **p** - your initial parameters (dict):

```python

p = {'w': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b': 0.0001}

```

And your model will be fitted automatically using numerical methods.
No need to analytically solve for the gradients!
It numerically estimates the gradients for you!

#### NOTE: LambdaML is still under active development.

#### Basic Usage

```python

import numpy as np
from lambda_model import LambdaClassifierModel

def synthesize_data():
    """Create fictitious data set consisting of two linearly-separable clusters"""
    np.random.seed(42)
    num_observations = 200

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations),
                                np.ones(num_observations)))
    
    return X,Y

def sine_activation(x):
    """Custom activation function."""
    return (np.sin(x / 30) + 1) / 2

def sine_regression(x,p):
    """Custom sine-based regression model."""
    signal = p['w'].dot(x) + p['b']
    return sine_activation(signal)

X,Y = synthesize_data()

# inital parameters
p = {'w': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b': 0.0001}

# create model
model = LambdaClassifierModel(f=sine_regression, p=p)

# fit the model
print('before:', model.compute_log_likelihood(X,Y))
model.fit(X,Y,n_iter=100)
print('after:', model.compute_log_likelihood(X,Y))

# predict classes
y_pred = model.predict(X)

# measure accuracy
print('Accuracy:', 1 - np.abs(Y - y_pred).mean())


```