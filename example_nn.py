import numpy as np
from lambda_model import LambdaClassifierModel

def synthesize_data():
    """Create fictitious data set consisting of two linearly-separable clusters"""
    np.random.seed(42)
    num_observations = 2000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations),
                                np.ones(num_observations)))
    
    return X,Y

def tanh_neuron(x,p,w_key,b_key):
    """Custom tanh-based regression model."""
    signal = p[w_key].dot(x) + p[b_key]
    return np.tanh(signal)

def hidden_layer(x, p):
    return [
        tanh_neuron(x,p,'w1','b1'),
        tanh_neuron(x,p,'w2','b2'),
    ]

def neural_network(x, p):
    activations = hidden_layer(x,p)
    signal = p['wf'].dot(activations) + p['bf']
    return (np.tanh(signal) + 1) / 2

X,Y = synthesize_data()

# inital parameters
p = {
    'w1': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b1': np.random.uniform(),
    'w2': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b2': np.random.uniform(),
    'wf': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'bf': np.random.uniform(),
}

# create model
model = LambdaClassifierModel(f=neural_network, p=p)

# fit the model
print('before:', model.compute_log_likelihood(X,Y))
model.fit(X,Y,n_iter=10)
print('after:', model.compute_log_likelihood(X,Y))

# predict classes
y_pred = model.predict(X)

# measure accuracy
print('Accuracy:', 1 - np.abs(Y - y_pred).mean())
