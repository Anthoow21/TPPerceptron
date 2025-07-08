# Rapport – TP1 : Perceptron simple

## 1. Introduction : Contexte et objectifs

L’objectif de ce TP était de comprendre comment fonctionne un perceptron simple, c’est-à-dire un modèle de neurone artificiel capable de résoudre des problèmes de classification linéaire.  
On devait l’implémenter nous-mêmes, voir comment il apprend avec les données, et surtout mettre en évidence ses limites. Ce TP sert aussi d’introduction avant de passer aux réseaux de neurones plus complexes (multi-couches).

## 2. Méthodes : Description des algorithmes implémentés

On a commencé par implémenter un perceptron simple, en suivant l’algorithme classique :  
- initialisation aléatoire des poids et biais,  
- propagation des données (produit scalaire + fonction d’activation),  
- mise à jour des poids en fonction de l’erreur avec la règle du perceptron.  

On a testé plusieurs fonctions d’activation pour voir leur impact sur l’apprentissage.  
On a aussi ajouté la possibilité de visualiser les frontières de décision, ce qui aide beaucoup à comprendre comment le modèle sépare les classes.

## 3. Résultats

### Tests sur fonctions logiques

On a testé notre perceptron sur plusieurs fonctions logiques (AND, OR).  
Résultat : ça marche bien pour les fonctions linéaires. Le modèle apprend vite et sépare correctement les données.  
Mais dès qu’on passe à XOR, ça ne fonctionne plus, peu importe le nombre d’itérations. C’est normal car XOR n’est pas linéairement séparable, donc le modèle n’a pas la capacité de le résoudre avec une seule couche.

#### Résultat du test sur AND : 

![Fonction AND](/img_tp1/test_AND_perceptron.png)

#### Résultat du test sur OR : 

![Fonction OR](/img_tp1/test_OR_perceptron.png)

#### Résultat du test sur XOR : 

![Fonction XOR](/img_tp1/test_XOR_perceptron.png)

Grâce à ce graphique, on se rend bien compte que XOR n'est pas linéairement séparable, et qu'il est donc impossible de résoudre ce problème avec une seule couche.

### Analyse de convergence

On a aussi testé différents taux d’apprentissage (η).  
- Si η est trop grand (genre 1), le modèle devient instable et ne converge pas.  
- Si η est trop petit (genre 0.0001), il apprend très lentement voire pas du tout.  
- Un bon compromis est autour de 0.01 ou 0.001 selon les cas.

#### Analyse de la convergence (graphique) : 

![Convergence perceptron](/img_tp1/analyse_convergence_perceptron.png)

Ce graphique permet de bien visualiser que lorsque le η est entre 0.01 ou 0.001, cela converge vers 0 pour le nombre d'erreurs.

### Évaluation sur données réelles

On a testé le perceptron sur des données plus "réalistes" comme un sous-ensemble du dataset Iris (en sélectionnant 2 classes et 2 features).  
Quand les classes sont bien séparées dans le plan, le modèle arrive à bien classifier.

#### Dataset iris binaire : 

![Iris binaire réelles](/img_tp1/dataset_iris_binaire_reelles.png)
![Iris binaire prédictions](/img_tp1/dataset_iris_binaire_predictions.png)

Ces deux graphiques montrent la comparaison entre les vraies classes (image 1) et les prédictions (image 2). On remarque que les deux sont identiques, ce qui montre que le perceptron apprend correctement.

#### Dataset iris multiclasse : 

![Iris multi-classe](/img_tp1/dataset_iris_multiclasse.png)

Sur le test multi-classe, on remarque que pour les setosas et les virginicas, le perceptron a bien apprit et que les 15 prédits sont bien les 15 vrais. Cependant, 6 des versicolors ont été prédit en tant que virginicas et 1 des versicolor a été prédi en tant que setosa. Cela signifie que les setosas ont des caractéristiques assez proches des deux autres classes, et/ou que le modèle est trop simple ou pas assez entraîné.

## 4. Discussion

### Limites du perceptron

La principale limite, c’est qu’il ne peut résoudre que des problèmes linéaires. S’il n’y a pas de frontière linéaire entre les classes, il ne peut pas faire mieux.  
Il est donc très limité sur des problèmes un peu plus complexes.

# Rapport TP2 – Réseau Multicouche et Backpropagation

## 1. Introduction

Dans ce TP, on s’est attaqué aux limites du perceptron simple en passant aux réseaux multicouches. L’objectif était de comprendre comment un MLP (Multi-Layer Perceptron) permet de résoudre des problèmes plus complexes comme XOR, qu’un perceptron classique ne peut pas gérer.

On a implémenté toute la mécanique : architecture, propagation avant, rétropropagation, mise à jour des poids... pour pouvoir tester tout ça sur des fonctions logiques et des datasets un peu plus réalistes.

## 2. Méthodes

### Architecture du réseau
On définit un réseau comme une suite de couches (entrée, cachées, sortie). Chaque couche est constituée de neurones avec poids, biais, fonction d'activation.

L’architecture est définie par une liste, ex :  `[2, 3, 1]` = 2 entrées, 1 couche cachée de 3 neurones, 1 neurone en sortie.

### Propagation avant
À chaque couche :
- On fait une combinaison linéaire `z = W.x + b`
- On applique une activation (sigmoid, tanh, relu, ...)

Les valeurs intermédiaires sont stockées pour la rétropropagation.

### Rétropropagation
On applique la règle de dérivation en chaîne :
- On part de la dernière couche avec l’erreur `y_pred - y_true`
- On remonte couche par couche en calculant le gradient des poids et biais
- On met à jour les paramètres en fonction du taux d’apprentissage

### Fonctions de coût
On utilise l'erreur quadratique moyenne :

```
loss = 0.5 * (y_pred - y_true)^2
```

L’accuracy est aussi calculée pour voir si le réseau converge bien.

## 3. Résultats

### Résolution du problème XOR

Le réseau multicouche arrive bien à résoudre le problème XOR, contrairement au perceptron simple. Il faut au minimum une couche cachée avec 2 neurones pour que ça fonctionne. Avec plus de neurones, ça converge plus vite et plus facilement (attention, trop en mettre n'est pas la solution non plus).
Une architecture que j'ai testé et qui marche est : `[2, 2, 1]`  (minimal). Les meilleures architectures que j'ai pu tester sont : `[2, 3, 1]` et `[2, 4, 1]`. En voulant mettre deux couches cachées (ex : `[2, 2, 2, 1]`), cela ne marchait pas bien (accuracy de 0.5).

### Tests sur datasets synthétiques et réels

Voici ce qu'on appelle une "surface de décision". Ca ressemble à ce que l'on avait fait pour le tp précédent avec la courbe de séparation, mais qui ne marchait pas pour XOR (pas linéairement séparable).

![Surface de décision XOR](/img_tp2/surface_decision_XOR.png)

Elle permet de visualiser la séparation des valeurs

<!-- TODO : faire des tests sur des datasets synthétiques et réels -->
<!-- Faire avec Iris -->

### Analyse de l'architecture

Dans mon cas, j'ai testé 4 types d'architectures principaux avec XOR : 
- `[2, 2, 1]` ->  2 neuronnes d'entrées, une couche de 2 neuronnes cachés et 1 neuronne de sortie. Avec cette architecture, la précision peut varier. L'accuracy varie entre 0.5 et 0.75 (1 dans des cas plus rare). De ce fait, elle n'est pas optimale mais permet d'approximer le problème XOR assez facilement (cas minimum).
- `[2, 3, 1]` -> 2 neuronnes d'entrées, une couche cachée de 3 neuronnes et 1 neuronne de sortie. Avec cette architecture, la précision est dans la plupart des cas 1 (ça m'est déjà arrivé d'avoir 0.5 ou 0.75 mais assez rare). Cela fait de cette architecture un bon choix pour le problème XOR.
- `[2, 4, 1]` -> 2 neuronnes d'entrées, une couche cachée de 4 neuronnes et 1 neuronne de sortie. Sur cette architecture, après avoir fait pas mal de test, je suis toujours tombé avec une accuracy de 1. De ce fait, cela semble être la meilleure architecture à privilégier pour ce problème.
- `[2, 3, 2, 1]` -> 2 neuronnes d'entrées, une couche cachée de 3 neuronnes, une couche cachée de 2 neuronnes et 1 neuronne de sortie. Ici, on obtient également une accuracy de 1 pour le problème XOR. Cependant, l'utilisation de deux couches cachées peut être un peu plus lourd que si on en avait qu'une seule. Pour un problème binaire tel que XOR, ce n'est pas forcément nécessaire et/ou utile d'utiliser ce type d'architecture.

Comme expliqué, les architectures `[2, 3, 1]` et `[2, 4, 1]` semblent être les plus performantes pour résoudre un problème XOR. Elles permettent toutes les deux d'avoir une accuracy de 1 (dans la plupart des cas). Cette accuracy peut varier. Cependant, l'architecture `[2, 4, 1]` semble être la plus performante car elle m'a toujours donnée une accuracy de 1.

Exemple de résultat pour le problème XOR : 

![Résultat sortie XOR](/img_tp2/resultat_sortie_XOR.png)

#### Courbe d’apprentissage

<!-- TODO : écrire le texte -->

Courbes de perte et d'accuracy comparée : 

![Courbe perte et accuracy](/img_tp2/courbe_perte_et_accuracy.png)

### 4. Discussion

#### Avantages des MLP

Les principaux avantages des MLP sont qu'ils permettent de résourde des problèmes non linéaires (comme le XOR) et qu'ils peuvent s'adapter à pas mal de type de données.

#### Inconvénients

Tout comme le perceptron simple vu au TP1, si on ne choisit pas bien le taux d'apprentissage, cela ne peut jamais converger ou avoir un comportement inattendu.

#### Sur-apprentissage

<!-- TODO : Faire courbe sur-apprentissage dans test_multicouche.py -->

### 5. Conclusion

<!-- Parler de la vidéo ludique pour MLP -->