from tests import test_couche, test_multicouche

if __name__ == '__main__':
    while True:
        print("\n=== Menu principal ===")
        print("1. Test une seule couche")
        print("2. Test multicouche")
        print("3. Quitter")

        choix = input("Choisissez une option (1-3) : ").strip()

        if choix == '1':
            print("\n=== Test du perceptron une seule couche ===")
            print("a. Propagation avant")
            print("b. Rétropropagation")
            choix = input("Choisissez une option (a/b) : ").strip()
            if choix == 'a':
                test_couche.test_forward()
            elif choix == 'b':
                test_couche.test_backward()

        elif choix == '2':
            print("\n=== Test du perceptron multicouche ===")
            print("a. Propagation avant")
            print("b. Rétropropagation")
            print("c. Test XOR")
            choix = input("Choisissez une option (a-c) : ").strip()
            if choix == 'a':
                test_multicouche.test_forward()
            elif choix == 'b':
                test_multicouche.test_backward()
            elif choix == 'c':
                test_multicouche.test_xor()

        elif choix == '3':
            break

        else:
            print("Option invalide. Veuillez choisir un numéro entre 1 et 3.")
