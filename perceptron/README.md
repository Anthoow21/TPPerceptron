# TP Réseaux de Neurones – Perceptron Simple

## Organisation du projet

Structure du projet :

perceptron/
├── main.py
├── src/
│ ├── activation.py
│ └── perceptron.py
├── tests/
│ └── test_perceptron.py
├── README.md
└── requirements.txt


## Utilisation

Installer les dépendances : 

```
pip install -r requirements.txt
```

Lancer le menu :

```
python main.py
```

## Réponses aux questions

### Exercice 2 : Questions d'analyse

**Pourquoi la fonction Heaviside pose problème pour l'apprentissage par gradient ?**  
Elle est pas dérivable, donc le gradient est inutilisable.

**Sigmoid vs Tanh ?**  
Sigmoid sort entre 0 et 1 -> bien pour y appartient à {0,1}.  
Tanh sort entre -1 et 1 -> mieux adapté quand y appartient à {-1,1}
En plus, tanh est centrée autour de 0 -> meilleure convergence.

**Pourquoi ReLU est populaire dans les réseaux profonds ?**  
C'est une fonction simple et rapide. Elle garde les gradients non nuls pour x > 0, ce qui aide à propager les infos dans les couches.

**Avantage du Leaky ReLU ?**  
Elle corrige le souci de Relu qui "meurt" quand x < 0. Avec Leaky relu, on a toujours un petit gradient même pour les négatifs.

---

### Exercice 3 : Questions sur le taux d’apprentissage

**Si η est trop grand ?**  
Ça fait un peu "n'importe quoi", l’algorithme diverge et l'apprentissage est instable.

**Si η est trop petit ?**  
C’est relativement lent, on met du temps à converger voir on ne converge pas du tout.

**Valeur idéale ?**  
Je n'ai pas trouvé de "valeure idéal", mais en testant avec un η aux alentours de 0.01, ça donne de bons résultat.

**Peut-on faire varier η au cours du temps ?**  
Oui, on pourrait le faire diminuer progressivement au fil des époques pour facilier l'apprentissage vers la fin.

---

### Exercice 6 : Test de XOR

**Constatation ?**  
Le perceptron échoue avec XOR.

**Pourquoi ?**  
XOR n’est pas linéairement séparable ce qui fait qu'on ne pourra rien faire avec un seul neuronne. Il faudra passer sur du multicouche comme évoquer dans la suite du TP  

---

### Exercice 8 : Convergence

**Quand η est très petit ?**  
Soit on apprent pas, soit très lentement (selon le nombre d'époque)

**Quand η est trop grand ?**  
C'est assez aléatoire dans les tests que j'ai pu faire. Parfois ça convergait, parfois non.

**Y a-t-il un η optimal ?**  
Cela dépend un peu des données, mais un η entre 0.1 et 0.01 est plutôt pas mal.

**Interaction avec η**  
Si les données ont un bruit "fort" ou mal séparées, un gros η amplifie les erreurs. Le choix du taux dépend donc aussi du bruit et de la distribution.

### Exercice 9 : Perceptron multi-classes

**Cohérence des prédictions**  
Si plusieurs perceptrons "président" sur un même point, on regarde les scores et celui qui a le plus haut score "gagne".

**Gestion des ambiguïtés** 
Même si tous les scores sont négatifs, il suffit de prendre la classe la moins mauvaise, celle qui a le score le plus élevé (même si il est négatif)

**Équilibrage** 
Je ne pense pas qu'elle le gère vraiment (j'ai fais quelque recherche mais je n'ai rien trouvé sur ça). Il faudrait équiliber les jeux d'entrainement pour régler ce soucis.



### Questions de réflexion et d'analyse

**Convergence**
Le perceptron converge uniquement si les données sont linéairement séparables.

**Initialisation**
Oui, ça peut jouer. Les poids de départ peuvent influencer la trajectoire de l’apprentissage et donc la solution finale.

**Taux d’apprentissage**
On peut le faire en "tatonnant". S’il est trop petit ça apprend lentement, s’il est trop grand ça diverge. En général, entre 0.01 et 0.1, ça fonctionne bien.

**Généralisation**
On regarde l’accuracy sur un jeu de test qui n’a pas servi à l’entraînement. Si elle est proche de celle de l'entraînement, alors ça généralise bien

**XOR revisité**
Un perceptron simple peut pas résoudre XOR. Il faut au moins une couche caché ou modifier les fonctionnalités pour rendre le problème linéairement séparable.

**Données bruitées**
Il va avoir du mal car le perceptron va chercher à corriger toutes les erreurs, même celles dues au bruit, donc il peut se tromper ou ne jamais converger.

**Classes déséquilibrées**
Il apprend surtout la classe majoritaire et peut complètement ignorer la classe minoritaire.Il faut donc équilibrer le dataset dans ce cas-là.

**Normalisation**
Oui, c’est mieux de normaliser les données. Ca met toutes les features sur la même échelle et ça aide l’apprentissage à être plus stable.