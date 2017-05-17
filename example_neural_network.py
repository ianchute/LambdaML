import numpy as np
import pandas as pd
from lambda_model import LambdaClassifierModel

def get_data():
    """Non-linearly separable data."""
    circles = pd.read_csv('data/circles.csv')
    return circles[['x','y']].values, circles['label'].values

def tanh_neuron(x,p,w_key,b_key):
    """Neuron with tanh activation."""
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
model.fit(X,Y,n_iter=150)
print('after:', model.compute_log_likelihood(X,Y))

# predict classes
y_pred = model.predict(X)

# measure accuracy
print('Accuracy:', 1 - np.abs(Y - y_pred).mean())
