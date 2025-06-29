import numpy as np
from src.multicouche import PerceptronMultiCouches

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

import numpy as np
from src.multicouche import PerceptronMultiCouches

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
