import numpy as np
from src.perceptron_multi import PerceptronMultiClasse
from src.dataset import (
    charger_donnees_iris_binaire,
    charger_donnees_iris_complete,
    visualiser_iris
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def test_iris_binaire():
    X, y = charger_donnees_iris_binaire()
    model = PerceptronMultiClasse(learning_rate=0.01)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"Accuracy binaire : {acc:.2f}")
    visualiser_iris(X, y, title="Iris binaire - vraies classes")
    visualiser_iris(X, y_pred, title="Iris binaire - prédictions")

def test_iris_multiclasse():
    X, y, noms = charger_donnees_iris_complete()
    evaluer_perceptron_multiclasse(X, y, noms)

def evaluer_perceptron_multiclasse(X, y, target_names=None, test_size=0.3, val_size=0.5):
    """
    Évalue le perceptron multi-classes avec une méthodologie rigoureuse
    """
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42)

    model = PerceptronMultiClasse(learning_rate=0.01)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    print("Accuracy (train):", np.mean(y_pred_train == y_train))
    print("Accuracy (val):", np.mean(y_pred_val == y_val))
    print("Accuracy (test):", np.mean(y_pred_test == y_test))

    print("\nClassification Report (test) :")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Matrice de confusion (test)")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.show()
