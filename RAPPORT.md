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

## Rapport TP2 – Réseau Multicouche et Backpropagation

### 1. Introduction

Dans ce TP, on s’est attaqué aux limites du perceptron simple en passant aux réseaux multicouches. L’objectif était de comprendre comment un MLP (Multi-Layer Perceptron) permet de résoudre des problèmes plus complexes comme XOR, qu’un perceptron classique ne peut pas gérer.

On a implémenté toute la mécanique : architecture, propagation avant, rétropropagation, mise à jour des poids... pour pouvoir tester tout ça sur des fonctions logiques et des datasets un peu plus réalistes.

### 2. Méthodes

#### Architecture du réseau
On définit un réseau comme une suite de couches (entrée, cachées, sortie). Chaque couche est constituée de neurones avec poids, biais, fonction d'activation.

L’architecture est définie par une liste, ex :  `[2, 3, 1]` = 2 entrées, 1 couche cachée de 3 neurones, 1 neurone en sortie.

#### Propagation avant
À chaque couche :
- On fait une combinaison linéaire `z = W.x + b`
- On applique une activation (sigmoid, tanh, relu, ...)

Les valeurs intermédiaires sont stockées pour la rétropropagation.

#### Rétropropagation
On applique la règle de dérivation en chaîne :
- On part de la dernière couche avec l’erreur `y_pred - y_true`
- On remonte couche par couche en calculant le gradient des poids et biais
- On met à jour les paramètres en fonction du taux d’apprentissage

#### Fonctions de coût
On utilise l'erreur quadratique moyenne :

```
loss = 0.5 * (y_pred - y_true)^2
```

L’accuracy est aussi calculée pour voir si le réseau converge bien.

---

### 3. Résultats

#### Résolution du problème XOR

Le réseau multicouche arrive bien à résoudre le problème XOR, contrairement au perceptron simple. Il faut au minimum **une couche cachée avec 2 neurones** pour que ça fonctionne. Avec plus de neurones, ça converge plus vite et plus facilement (attention, trop en mettre n'est pas la solution non plus).
Une architecture que j'ai testé et qui marche est : `[2, 2, 1]`  (minimal). Les meilleures architectures que j'ai pu tester sont : `[2, 3, 1]` et `[2, 4, 1]`. En voulant mettre deux couches cachées (ex : `[2, 2, 2, 1]`), cela ne marchait pas bien (accuracy de 0.5).

#### Tests sur datasets synthétiques et réels


#### Analyse de l'architecture



#### Courbes d’apprentissage

### 4. Discussion

#### Avantages des MLP


#### Inconvénients


#### Sur-apprentissage

### 5. Conclusion
