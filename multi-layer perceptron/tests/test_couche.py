import numpy as np
from src.couche import CoucheNeurones

def test_forward():
    couche = CoucheNeurones(n_input=3, n_neurons=2, activation='relu', learning_rate=0.01)
    X = np.array([[1, 2], [3, 4], [5, 6]])
    output = couche.forward(X)
    print("Sortie après propagation avant :\n", output)

def test_backward():
    couche = CoucheNeurones(n_input=2, n_neurons=1, activation='sigmoid', learning_rate=0.1)
    X = np.array([[0.5], [0.8]])
    output = couche.forward(X)
    y_true = np.array([[1]])
    loss_grad = output - y_true
    grad = couche.backward(loss_grad)
    print("Gradient d'entrée retourné :\n", grad)
