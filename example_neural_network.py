import numpy as np
import pandas as pd
from lambda_model import LambdaClassifierModel

def get_data():
    """Non-linearly separable data."""
    circles = pd.read_csv('data/circles.csv')
    return circles[['x','y']].values, circles['label'].values

def neuron(x,p,w_key,b_key):
    """Neuron with ReLU activation."""
    signal = p[w_key].dot(x) + p[b_key]
    return np.max([0, signal])

def hidden_layer_1(x, p):
    return [
        neuron(x,p,'w1','b1'),
        neuron(x,p,'w2','b2'),
    ]

def hidden_layer_2(x, p):
    return [
        neuron(x,p,'w3','b3'),
        neuron(x,p,'w4','b4'),
    ]

def neural_network(x, p):
    activations_1 = hidden_layer_1(x,p)
    activations_2 = hidden_layer_2(activations_1,p)
    signal = p['wf'].dot(activations_2) + p['bf']
    return (np.tanh(signal) + 1) / 2

X,Y = get_data()

# inital parameters
p = {
    'w1': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b1': np.random.uniform(),
    'w2': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b2': np.random.uniform(),

    'w3': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b3': np.random.uniform(),
    'w4': np.array([np.random.uniform() for i in range(len(X[0]))]),
    'b4': np.random.uniform(),

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
