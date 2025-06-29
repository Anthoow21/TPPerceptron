import numpy as np
import matplotlib.pyplot as plt
from src.perceptron import PerceptronSimple
from src.perceptron_multi import PerceptronMultiClasse

def generer_donnees_separables(n_points=100, noise=0.1):
    np.random.seed(42)
    X1 = np.random.randn(n_points, 2) + np.array([2, 2])
    y1 = np.ones(n_points)
    X2 = np.random.randn(n_points, 2) + np.array([-2, -2])
    y2 = -np.ones(n_points)
    X1 += noise * np.random.randn(n_points, 2)
    X2 += noise * np.random.randn(n_points, 2)
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def visualiser_donnees(X, y, w=None, b=None, title="Données"):
    plt.figure(figsize=(8, 6))
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe -1')
    if w is not None and b is not None:
        x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        x2 = -(w[0] * x1 + b) / w[1]
        plt.plot(x1, x2, color='green', label='Droite de séparation', linewidth=2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]):
    plt.figure(figsize=(12, 8))
    max_epochs = 100
    for lr in learning_rates:
        perceptron = PerceptronSimple(learning_rate=lr)
        errors = perceptron.fit(X, y, max_epochs=max_epochs, return_errors=True)
        epochs = np.arange(len(errors))
        plt.plot(epochs, errors, label=f'lr={lr}', marker='o', linestyle='-', alpha=0.7)

    plt.xlabel('Époque')
    plt.ylabel("Nombre d'erreurs")
    plt.title("Convergence pour différents taux d'apprentissage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def test_AND():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])
    perceptron = PerceptronSimple()
    perceptron.fit(X, y)
    print("AND Predictions:", perceptron.predict(X))
    print("AND Score:", perceptron.score(X, y))
    visualiser_donnees(X, y, perceptron.weights, perceptron.bias, title="Fonction AND")

def test_OR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, 1])
    perceptron = PerceptronSimple()
    perceptron.fit(X, y)
    print("OR Predictions:", perceptron.predict(X))
    print("OR Score:", perceptron.score(X, y))
    visualiser_donnees(X, y, perceptron.weights, perceptron.bias, title="Fonction OR")

def test_XOR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])
    perceptron = PerceptronSimple()
    perceptron.fit(X, y)
    print("XOR Predictions:", perceptron.predict(X))
    print("XOR Score:", perceptron.score(X, y))
    visualiser_donnees(X, y, perceptron.weights, perceptron.bias, title="Fonction XOR")

def test_multiclasse():
    # Exemple simple : 3 classes bien séparées
    np.random.seed(42)
    n = 50
    X1 = np.random.randn(n, 2) + np.array([5, 0])
    y1 = np.zeros(n)

    X2 = np.random.randn(n, 2) + np.array([0, 5])
    y2 = np.ones(n)

    X3 = np.random.randn(n, 2) + np.array([-5, 0])
    y3 = np.full(n, 2)

    X = np.vstack([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    model = PerceptronMultiClasse(learning_rate=0.01)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"Score multi-classe : {acc:.2f}")

    # Visualisation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", label="vrai")
    plt.title("Données multi-classe (couleur = vraie classe)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab10", label="préd")
    plt.title("Prédictions du perceptron multi-classe")
    plt.grid(True)
    plt.show()
