# TP Réseaux de Neurones – Perceptron Multi-Couches

## Organisation du projet

Structure du projet :

mutli-layer perceptron/
├── main.py
├── src/
│ ├── activation.py
│ ├── couche.py
│ └── multicouche.py
├── tests/
│ ├── test_couche.py
│ └── test_multicouche.py
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

---

### Exercice 1.1 - Analyse théorique

**Que signifie concrètement le théorème d'approximation universelle ?**  
Qu’un perceptron mutli-couche avec une seule couche cachée (et assez de neurones) peut approximer n’importe quelle fonction continue.

**Ce théorème garantit-il qu'on peut toujours trouver les bons poids ?**  
Non, il dit juste que la solution existe, pas qu’on va la trouver facilement.

**Différence entre "pouvoir approximer" et "pouvoir apprendre" ?**  
"Approximable" = c’est possible avec les bons poids.  
"Apprendre" = c’est les trouver en pratique, via de l'entraînement.

**Pourquoi beaucoup de couches cachées en pratique ?**  
Ça permet d’avoir plus de flexibilité, de profondeur, et de mieux généraliser avec moins de neurones par couche.

**Autres approximateurs vus au lycée ?**  
Il me semble qu'on avait déjà vu les polynômes de Taylor, mais je n'en ai pas beaucoup de souvenir. J'étais en STI2D et je ne pense pas que l'on ait vu autre chose. 

---

### Exercice 1.2 - Phrase à expliquer

**"Le théorème d’approximation universelle affirme qu’un réseau profond peut exactement retrouver les données d’entraînement."**  
Oui, il peut "coller" parfaitement aux données, mais ça veut pas dire qu’il va bien généraliser derrière.

---

### Exercice 2.2 - Propagation avant

**Rôle de la propagation avant ?**  
Faire passer les données à travers toutes les couches pour obtenir une prédiction à la fin.

**Pourquoi une forme matricielle ?**  
C’est plus rapide, plus propre, et ça permet de faire tourner le réseau sur plusieurs exemples en même temps.

---

### Exercice 3.1 - Test XOR

**Le réseau arrive-t-il à résoudre XOR ?**  
Oui, dès qu’on met au moins une couche cachée avec au moins 3 neuronnes (avec 2 neuronnes cachés, ça ne le résous pas tout le temps)

**Nombre de neurones cachés ?**  
3 neuronnes cachés peuvent suffire. Si on en met plus, on convergera naturellement plus vite. Cependant, il est préférable de mettre 1 couche caché avec 8 neurones que 2 couches cachées avec 4 neuronnes chacune.

**Plusieurs couches cachées ?**  
Comme expliqué plus tôt, ça n'a pas vraiment d'intérêt d'utiliser plusieurs couches cachées pour un problème comme XOR qui est relativement simple (problème binaire). Pour des problèmes plus complexes, cela pourrait être plus intéressant.

**L'initialisation des poids a-t-elle une influence ?**  
Oui, selon les valeurs, ça peut ralentir ou empêcher l'apprentissage.