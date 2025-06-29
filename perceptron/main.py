from tests import test_perceptron

if __name__ == '__main__':
    while True:
        print("\n=== Menu Principal ===")
        print("1. Visualiser les données avec la droite de séparation")
        print("2. Analyser la convergence")
        print("3. Tester une fonction logique (AND, OR, XOR)")
        print("4. Tester le perceptron multi-classe")
        print("5. Tester sur le dataset Iris (binaire ou multi-classe)")
        print("6. Quitter")

        choix = input("Choisissez une option (1-6) : ").strip()

        if choix == '1':
            X, y = test_perceptron.generer_donnees_separables()
            perceptron = test_perceptron.PerceptronSimple()
            perceptron.fit(X, y)
            test_perceptron.visualiser_donnees(X, y, perceptron.weights, perceptron.bias, title="Visualisation Données")

        elif choix == '2':
            X, y = test_perceptron.generer_donnees_separables()
            test_perceptron.analyser_convergence(X, y)

        elif choix == '3':
            print("\n--- Fonctions Logiques Disponibles ---")
            print("a. AND")
            print("b. OR")
            print("c. XOR")
            sous_choix = input("Choisissez une fonction à tester (a/b/c) : ").strip().lower()

            if sous_choix == 'a':
                test_perceptron.test_AND()
            elif sous_choix == 'b':
                test_perceptron.test_OR()
            elif sous_choix == 'c':
                test_perceptron.test_XOR()
            else:
                print("Choix non reconnu. Veuillez sélectionner a, b ou c.")
        elif choix == '4':
            test_perceptron.test_multiclasse()
        elif choix == '5':
            print("\n--- Iris Dataset ---")
            print("a. Version binaire")
            print("b. Version complète")
            sous_choix = input("Choisissez une version (a/b) : ").strip().lower()

            if sous_choix == 'a':
                from tests.test_iris import test_iris_binaire
                test_iris_binaire()
            elif sous_choix == 'b':
                from tests.test_iris import test_iris_multiclasse
                test_iris_multiclasse()
            else:
                print("Choix non reconnu.")
        elif choix == '6':
            break

        else:
            print("Option invalide. Veuillez choisir un numéro entre 1 et 6.")
