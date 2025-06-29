import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z<0,0,1)
        elif self.name == "sigmoid":
            return 1/(1+np.exp(-z))
        elif self.name == "tanh":
            return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        elif self.name == "relu":
            return np.where(z<0,0,z)
        elif self.name == "leaky_relu":
            return np.where(z<0,self.alpha*z,z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            epsilon=1e-6
            return np.where(np.abs(z) < epsilon, 1/epsilon, 0)
        elif self.name == "sigmoid":
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif self.name == "tanh":
            return 4/np.square(np.exp(z)+np.exp(-z))
        elif self.name == "relu":
            return np.where(z>0,1,0)
        elif self.name == "leaky_relu":
            return np.where(z>0,1,self.alpha)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
