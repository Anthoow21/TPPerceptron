import numpy as np
from tqdm import tqdm
from .activation import ActivationFunction

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100, return_errors=False):
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0
        activation = ActivationFunction("tanh")
        errors = []

        for e in range(max_epochs):
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]

                y_pred = activation.apply(np.dot(self.weights, x) + self.bias)
                d = y_true - y_pred

                if d != 0:
                    self.weights += self.learning_rate * x * d
                    self.bias += self.learning_rate * d

            if return_errors:
                pred = self.predict(X)
                errors.append(np.sum(pred != y))

        return errors if return_errors else None

    def predict(self, X):
        activation = ActivationFunction("tanh")
        return np.array([activation.apply(np.dot(self.weights, x) + self.bias) for x in X])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)