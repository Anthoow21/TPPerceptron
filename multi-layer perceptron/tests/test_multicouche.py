import numpy as np
from src.multicouche import PerceptronMultiCouches
import matplotlib.pyplot as plt

def test_forward():
    # === Choix de la fonction logique ===
    # AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    # # OR
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0], [1], [1], [1]])

    model = PerceptronMultiCouches(architecture=[2, 3, 1], learning_rate=0.1, activation='sigmoid')
    output = model.forward(X)

    print("=== Test Forward ===")
    print("Entrée :")
    print(X)
    print("Sortie après propagation avant :")
    print(output)

def test_backward():
    print("\n=== Test Rétropropagation ===")

    # === Choix de la fonction logique ===
    # AND
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])

    # # OR
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0], [1], [1], [1]])

    model = PerceptronMultiCouches(architecture=[2, 2, 1], learning_rate=0.05, activation='sigmoid')

    model.fit(X, y, epochs=1000, verbose=True)

    # Prédictions finales
    y_pred = model.predict(X)
    print("\nPrédictions finales :")
    for i, (x, pred) in enumerate(zip(X, y_pred)):
        print(f"  Entrée : {x} => Prédiction : {pred[0]:.4f} / Attendu : {y[i][0]}")

def test_xor():
    """
    Test du réseau multicouches sur le problème XOR
    """
    # Données XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    print("\n=== Test XOR (réseau multicouche) ===")
    print("Données d'entrée :")
    print(X_xor)
    print("Sorties attendues :")
    print(y_xor.flatten())

    # Architectures à tester
    architectures = [
        [2, 2, 1],     # Minimal
        [2, 3, 1],     # Légèrement plus complexe
        [2, 4, 1],     # Peut résoudre XOR correctement
        [2, 2, 2, 1],  # Deux couches cachées
    ]

    for arch in architectures:
        print(f"\n--- Architecture testée : {arch} ---")

        # Création et entraînement
        mlp = PerceptronMultiCouches(architecture=arch, learning_rate=0.5, activation='sigmoid')
        mlp.fit(X_xor, y_xor, epochs=1000, verbose=False)

        # Prédictions
        y_pred = mlp.predict(X_xor)
        print("Prédictions :")
        for i in range(len(X_xor)):
            print(f"  Entrée : {X_xor[i]} => Prédiction : {y_pred[i][0]:.4f} / Attendu : {y_xor[i][0]}")

        # Accuracy finale
        acc = mlp.compute_accuracy(y_xor, y_pred)
        print(f"Accuracy : {acc:.2f}")

def test_learning_curves():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    model = PerceptronMultiCouches([2,3,1], learning_rate=0.5, activation='sigmoid')
    model.fit(X, y, epochs=500, verbose=False)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(model.history['loss'], label="Loss")
    plt.title("Courbe de perte")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(model.history['accuracy'], label="Accuracy", color='orange')
    plt.title("Courbe d'accuracy")
    plt.xlabel("Époques")
    plt.ylabel("Accuracy")
    plt.grid()

    plt.tight_layout()
    plt.show()

def test_decision_surface():
    from matplotlib.colors import ListedColormap

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    model = PerceptronMultiCouches([2,4,1], learning_rate=0.5, activation='sigmoid')
    model.fit(X, y, epochs=1000, verbose=False)

    h = 0.01
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']), alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), edgecolor='k', cmap='bwr')
    plt.title("Surface de décision (XOR)")
    plt.grid()
    plt.show()

def test_compare_architectures():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    architectures = [[2,2,1],[2,3,1],[2,4,1],[2,3,2,1]]

    for arch in architectures:
        model = PerceptronMultiCouches(arch, learning_rate=0.5, activation='sigmoid')
        model.fit(X, y, epochs=1000, verbose=False)
        acc = model.compute_accuracy(y, model.predict(X))
        print(f"Architecture {arch} => Accuracy : {acc:.2f}")

def test_surapprentissage():
    from sklearn.model_selection import train_test_split

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

    model = PerceptronMultiCouches([2,6,1], learning_rate=0.5, activation='sigmoid')
    model.fit(X_train, y_train, X_val, y_val, epochs=500, verbose=False)

    plt.plot(model.history['loss'], label='Train loss')
    plt.plot(model.history['val_loss'], label='Val loss')
    plt.title("Courbe de sur-apprentissage")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
