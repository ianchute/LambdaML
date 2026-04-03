import numpy as np
import pandas as pd
from lambda_model import LambdaClassifierModel

np.random.seed(1)

def get_data():
    """Non-linearly separable data."""
    circles = pd.read_csv('data/circles.csv')
    return circles[['x','y']].values, circles['label'].values

def neuron(x,p,w_key,b_key):
    """Exponential Linear Unit."""
    signal = p[w_key].dot(x) + p[b_key]

    if signal >= 0:
        return signal
    else:
        return 0.01 * (np.exp(signal) - 1)

def hidden_layer(x, p):
    return [
        neuron(x,p,'w1','b1'),
        neuron(x,p,'w2','b2'),
    ]

def neural_network(x, p):
    activations = hidden_layer(x,p)
    signal = p['wf'].dot(activations) + p['bf']
    return (np.tanh(signal) + 1) / 2

X,Y = get_data()

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
model.fit(X,Y,n_iter=200)
print('after:', model.compute_log_likelihood(X,Y))

# predict classes
y_pred = model.predict(X)

# measure accuracy
print('Accuracy:', 1 - np.abs(Y - y_pred).mean())
