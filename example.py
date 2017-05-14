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
