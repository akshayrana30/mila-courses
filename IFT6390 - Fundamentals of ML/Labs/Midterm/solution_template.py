import numpy as np

# %%
'''
# Plain Python
# Python simple
'''

# %%
'''
### Ones at the End
Given an int list X as input, return the same list, but with all the "1" at the end (i.e. all
the elements exactly equal to 1), and the rest of the list must be in reverse order of occurrence.

### Uns à la fin
Étant donné une liste d'entiers en entrée, renvoyer la même liste, mais avec tout les "1" à la fin
(c'est à dire tout les éléments exactement égaux à 1), et le reste de la liste doit être en ordre 
d'apparition inversé.
 
* [1, 1, 1] -> [1, 1, 1]
* [5, 9, -12, 3] -> [3, -12, 9, 5]
* [1, 2, 3, 1] -> [3, 2, 1, 1]
* [1, 5, 2, 1, 3, 1] -> [3, 2, 5, 1, 1, 1]
 '''


# %%
def ones_at_the_end(x):
    """
    :param x: python int list
    :return: python int list
    """
    pass


# %%
'''
### Final Position
Given a string of instructions 'up' 'down' 'left' 'right', return the end position of an agent
starting at (0,0). The first coordinate is the x-axis ('left' decreases its
value of 1 and 'right' increases its value of 1), and the second one is the y-axis ('down'
decreases its value of 1 and 'up' increases its value of 1).
(Hint : if X is a python string, X.split() will return the list of its words)

### Position Finale
Étant donné une chaîne de caractères formée d'instructions 'up' (haut), 'down' (bas),
'left' (gauche), 'right' (droite), retourner la position finale d'un agent qui commence
à la position (0,0). La première coordonnée est l'axe des abcisses ('left' decrémente sa
valeur de 1 et 'right' augmente sa valeur de 1), et la seconde coordonnée est l'axe des
ordonnées ('down' décremente sa valeur de 1 et 'up' augmente sa valeur de 1).
(Indice : si X est une chaîne de caractères python, X.split() la retournera sous forme d'une
liste de mots)

* "right right" -> (2, 0)
* "right left up" -> (0, 1)
* "down down left right up down" -> (0, -2)
'''


# %%
def final_position(instructions):
    """
    :param instructions: string
    :return: int tuple
    """
    pass


# %%
'''
### Steps to One
Let f be the following operation on integers :
if i is even, f(i) = i / 2
if i is odd, f(i) = 3i + 1
Let us now consider the algorithm that applies recursively operation f to an input i
until it reaches i=1 (we assume it will always do, eventually).
e.g. for i=7 we obtain the following iterations :
7 -> 22 -> 11 -> 34 -> 17 -> 52 -> 26 -> 13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
Implement a function which, given an int i as input, returns the number of steps required
to reach 1 when starting from i using this algorithm (i.e. number of times f is applied)

### Nombre d'Étapes Jusqu'à Un
Soit f l'opération sur les entiers suivante :
si i est pair, f(i) = i / 2
si i est impair, f(i) = 3i + 1
Considérons maintenant l'algorithme qui applique recursivement l'opération f à une entrée i
jusqu'à atteindre i=1 (Nous supposons que ce sera toujours le cas au bout d'un certain temps).
e.g. pour i=7 nous obtenons les étapes suivantes :
7 -> 22 -> 11 -> 34 -> 17 -> 52 -> 26 -> 13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
Implémentez une fonction qui, étant donné un entier i en entrée, retourne le nombre d'étapes
nécessaires pour atteindre 1 en partant de i en utilisant cet algorithme (i.e. le nombre de
fois où f est appliquée)

* 1 -> 0
* 8 -> 3
* 3 -> 7
* 7 -> 16 (above example / exemple ci-dessus)
'''


# %%
def steps_to_one(i):
    """
    :param i: int
    :return:  int
    """
    pass


# %%
'''
### Find Bins (BONUS FOR IFT3395 - UNDERGRAD)
Given a list of k * h different floats, return a list of k+1 floats in increasing order that
form bins (a bin is the interval between two consecutive floats of the output)
with exactly h input floats strictly inside each bin. In other words, if the output is
[x1, x2, ..., x(k+1)], there should be exactly h input float in ]x1, x2[, h input floats
in ]x2, x3[, ..., h input floats in ]xk, x(k+1)[
Solutions are not unique, but any valid output is accepted.

### Histogrammes (BONUS POUR IFT3395 - BAC)
Étant donnée une liste de k * h nombres réels distincts, retourner une liste de k+1 nombres
réels en ordre croissant qui forment des intervalles (entre deux nombre réels consécutifs)
avec exactement h nombre réels d'entrée strictement contenu dans chaque interval. En d'autres
termes, si la sortie est [x1, x2, ..., x(k+1)], il doit y avoir exactement h entrées dans
]x1, x2[, h entrées dans ]x2, x3[, ..., h entrées dans ]xk, x(k+1)[
Les solutions ne sont pas uniques, mais toute réponse valide sera acceptée.

* [2, 3, 4, 2.2], k=2 -> [1, 2.5, 5] or [-1000, 2.21, 1000] or ...
* [4, 3, 8, 6], k=4   -> [2.5, 3.5, 4.5, 6.5, 8.5] or [-10, 3.1, 4.9, 6.65, 100] or ...
'''


# %%
def find_bins(input_list, k):
    """
    :param input_list: list of k*h floats
    :param k: int
    :return: list of k+1 floats
    """
    pass


# %%
'''
# Numpy
'''

# %%
'''
### Even Odd Ordered
Given a 1-D array of int, return a 1-D array where even numbers are at the beginning and the
odd numbers are at the end, and these numbers are arranged in the order of occurrence

### Pairs et Impairs Ordonnés
Étant donné un tableau 1D d'entiers, retourner un tableau 1D où les entiers pairs sont au début
et les entiers impairs sont à la fin, dans le même ordre d'apparition qu'en entrée

* [1, 2, -3, 4, 7, 4, -6, 3, -1] -> [2, 4, 4, -6, 1, -3, 7, 3, -1]
* [-5, -4, -3, -2, -1, 0] -> [-4, -2, 0, -5, -3, -1]
'''


# %%
def even_odd_ordered(X):
    """
    :param X: np.array of shape (n,)
    :return: np.array of shape (n,)
    """
    pass


# %%
'''
### Dataset Normalization
Implement a function that standardize a dataset:
given an input matrix X of size n x (d+1), where the first d columns are the features and the
last one the target,
return a data matrix of the same shape where the data has been normalized (mean 0 for each
feature and stdv 1).
Note that the last column of the input array is preserved in the returned output

### Normalisation de Jeu de Données
Implémentez une fonction qui standardise un jeu de données:
étant donné en entrée une matrice X de taille n x (d+1), dont les d premières colonnes sont
les attributs et la dernière est l'objectif,
retourner une matrice de données de la même forme où les données ont été normalisées (chaque
attribut a une moyenne de 0 et une deviation standard de 1).
Remarque : la dernière colonne du tableau d'entrée est préservée dans le tableau de sortie

Input Array / Tableau d'entrée:
[[1., -1., 2., 1],
 [2., 0., 0., 2],
 [0., 1., -1., 3]]

Output Array / Tableau de sortie:        
[[0., -1.22474486, 1.3363062, 1],
[1.22474486, 0., -0.26726124, 2],
[-1.22474486, 1.22474486, -1.06904496, 3]]
'''


# %%
def data_normalization(X):
    """
    :param X: np.array of shape n x (d+1)
    :return: np.array of shape n x (d+1)
    """
    pass


# %%
'''
### Entropy of a Valid Discrete Probability Distribution
Find if a discrete probability distribution is valid. If valid, compute the entropy of the
distribution else return `None`
- A discrete probability distribution is valid if its entries are positive and sum to 1
- The entropy of a discrete probability distribution is defined as:
$entropy(p) = - \sum_{i=1}^N ( p_i * log_2(p_i))$
- if $p_i=0$ for some $i$, we adopt the convention $0*log_2(0) = 0$
Note: You are required to use the base of the logarithm as 2.

### Entropie d'Une Distribution de Probabilité Discrète Valide
Déterminer si une distribution de probabilité discrète est valide. Si elle est valide, calculer l'entropie de la
distribution sinon retourner `None`
- Une distribution de probabilité discrète est valide si ses valeurs sont positives et ont une somme de 1
- L'entropie d'une distribution de probabilité discrète est definie comme:
$entropy(p) = - \sum_{i=1}^N ( p_i * log_2(p_i))$
- si $p_i=0$ pour un certain $i$, nous adoptons la convention $0*log_2(0) = 0$
Remarque: Vous devez utiliser le logarithme de base 2.

* [0.6, 0.1, 0.25, 0.05] -> 1.490468  (valid probability distribution, so we return entropy / distribution
de probabilité valide, alors on retourne l'entropie)
* [0.5, 0.1, 0.25] -> None   (sum of the distribution is not one / la somme de la distribution n'est pas un)
* [0.3, 0.75, -0.3, 0.25] -> None   (all probability values of the distribution are not positive / certaines valeurs
ne sont pas positives)
'''


# %%
def entropy(p):
    """
    :param p: np.array of shape (n,)
    :return: float or None
    """
    pass

# %%
'''
### Heavyball Optimizer (BONUS FOR IFT3395 - UNDERGRADS)
In this question, we will be implementing the widely used  ‘Heavyball’ optimizer.
We want to find the minimum value of some function with parameter $x$ of size (n,). 
We update $x$ using gradient descent and a _momentum_ term.
To be clear, at iteration $k$, let the gradient of the function at $x_{k}$ be $g_{k}$.
We update the value of $x$ with the formula:
$$x_{k+1} = x_{k} - \alpha * g_{k} + \beta * (x_{k} - x_{k-1}), $$
where $\alpha$ and $\beta$ are hyperparameters of the algorithm. 

The function below takes as argument

* x the initial parameter $x_1$
* inputs = $[g_{1}, g_{2},..,g_{t}]$  a list of precomputed gradients
* $\alpha$ and $\beta$

We ask you to return the final parameter $x_{t+1}$, under the assumption that $x_{0}$ is a zero vector.

e.g.
Take $x = [1.5, 2.0]$, `inputs` = $[[1.5, 2.0]]$, $\alpha = 0.9$, $\beta = 0.1$. 

Then we have: $x_{0} = [0., 0.], x_{1} = [1.5, 2.0]$.

The length of the list inputs is 1, $g_{1} = [1.5, 2.0]$,
so we will only perform one iteration of the above loop

Now, let's compute the value for $x_{2}$
$$ x_{2} = [1.5, 2.0] - 0.9 * [1.5, 2.0] + 0.1 * ([1.5, 2.0] - [0., 0.]) $$
$$ x_{2} = 0.1 * [1.5, 2.0] + 0.1 * [1.5, 2.0] $$
$$ x_{2} = [0.3, 0.4] $$

return $x_{2} = [0.3, 0.4]$

----

### Optimiseur de Heavyball (BONUS POUR IFT3395 - BAC)
Vous allez implementer l'un des algorithmes d'optimisation les plus utilises en apprentissage automatique.
On veut trouver le minimum d'une fonction prenant un parametre $x$ de taille (n,).
On mets a jour $x$ avec un pas de gradient plus un terme de _moment_. 
Clarifions: a l'etape $k$, on a acces a $g_k$, le gradient de la fonction en $x_k$.
Alors on mets a jour le parametre $x$ avec la formule:
$$x_{k+1} = x_{k} - \alpha * g_{k} + \beta * (x_{k} - x_{k-1}),$$
ou $\alpha$ et $\beta$ sont des hyperparametres de l'algorithme.

La fonction ci-dessous prend les arguments

* x le parametre initial $x_1$
* inputs = $[g_{1}, g_{2},..,g_{t}]$ une liste de gradient pre-calcules.
* $\alpha$ et $\beta$

On vous demande de retourner le parametre final $x_{t+1}$, sous l'hypothese que $x_0=0$.

Par exemple, prenons $x = [1.5, 2.0]$, `inputs` = $[[1.5, 2.0]]$, $\alpha = 0.9$, $\beta = 0.1$. 

Dans ce cas, $x_{0} = [0., 0.], x_{1} = [1.5, 2.0]$.

La list inputs est de longueur 1, $g_{1} = [1.5, 2.0]$,
donc nous allons aire une seule iteration de la boucle.

On applique la formule
$$ x_{2} = [1.5, 2.0] - 0.9 * [1.5, 2.0] + 0.1 * ([1.5, 2.0] - [0., 0.]) $$
$$ x_{2} = 0.1 * [1.5, 2.0] + 0.1 * [1.5, 2.0] $$
$$ x_{2} = [0.3, 0.4] $$

On renvoie $[0.3, 0.4]$.
'''


# %%
def heavyball_optimizer(x, inputs, alpha=0.9, beta=0.1):
    """
    :param x: np.array of size (n,)
    :param inputs: a list of np.arrays of size (n,)
    :return: np.array of size (n,)
    """
    pass


# %%
''' 
### Machine Learning : Nearest Centroid Classifier

You are to implement a nearest centroid classifier for N points in D dimensions
with K classes. This is a simple classifier that looks at all the different
points in your data and comes up with K centroid points, one for each class.
A centroid point is the average of all the points of that class. E.g. In one
dimension for some class that has points 0, 1, 5, the centroid point would be
at 2.

For a new point, the classifier predicts the class whose centroid point is
closest to the new point. Use the L2 distance metric (sum of squared distances
in each dimension). In case of a tie, the classifier predicts the class with the
smaller number (e.g. class 0 over class 1). 
If there are k classes, then the label y take its value in [0, 1, ..., k-1].  

We provide the framework, you must fill in the methods. We've also provided
some code for you to be able to test your classifier. We've given a very basic
test case but it is up to you to figure out what is the correct performance on
this test case.

### Apprentissage Automatique : Classificateur du Centroïde le Plus Proche

Vous devez implémenter un classificateur du centroïde le plus proche pour
N points en D dimensions avec K classes. C'est un classificateur simple qui
regarde tout les différents points dans les données et calcule K centroïdes, un pour
chaque classe. Le centroïde d'une classe est la moyenne de tout les points de
cette classe. E.g. En une dimension, une classe dont les points seraient 0, 1, 5
aurait pour centroïde 2.

Étant donné un nouveau point, le classificateur lui prédit la classe dont le
centroïde est le plus proche de ce point. Utiliser la distance L2 (somme des
carrés des distances dans chaque dimension). En cas d'égalité, le classificateur
prédit la classe avec le plus petit nombre (e.g. la classe 0 plutôt que 1).
Si il y a k classes, alors les valeurs possibles du label y sont [0, 1, ..., k-1].

Nous fournissons le squelette, vous devez compléter les fonctions. Nous avons
aussi fourni un code pour que vous puissiez tester votre classificateur. Nous
avons fourni un exemple de test très simple mais c'est à vous de trouver quelle
est la performance attendue sur cet exemple.
'''


# %%
class NearestCentroidClassifier:
    def __init__(self, k, d):
        """Initialize a classifier with k classes in dimension d
        Initialise un classificateur avec k classes en dimension d

        :param k: int
        :param d: int
        """
        self.k = k
        self.centroids = np.zeros((k, d))

    def fit(self, X, y):  # question A
        """For each class k, compute the centroid and store it in self.centroids[k]
        Pour chaque class k, calcule le centroïde et l'enregistre dans self.centroids[k]

        :param X: float np.array of size N x d (each row is a data point / chaque ligne est un point)
        :param y: int np.array of size N (class of each data point / classe de chaque point)
        """

        # self.centroids[k] =
        pass

    def predict(self, X):  # question B
        """For each data point in the input X, return the predicted class
        Pour chaque point de l'entrée X, retourne la classe prédite

        :param X: float np.array of size N x d (each row is a data point / chaque ligne est un point)
        :return: int np.array of size N (predicted class of each data point / classe prédite pour chaque point)
        """

        # return predictions
        pass

    def score(self, X, y):  # question C
        """Compute the average accuracy of your classifier on the data points of input
        X with true labels y. That is, predict the class of each data point in X, and
        compare the predictions with the true labels y. Return how often your classifier
        is correct.
        Calcule la précision moyenne de votre classificateur sur les points de l'entrée X
        avec les vraies classes y. C'est à dire, prédisez la classe de chaque point de X,
        et comparez les prédictions avec les vraies classes y. Retourner avec quelle
        fréquence votre classificateur est correct.

        :param X: loat np.array of size N x d (each row is a data point / chaque ligne est un point)
        :param y: int np.array of size N (true class of each data point / vraie classe de chaque point)
        :return: float in [0,1] (success rate / taux de succès)
        """

        # return score
        pass


# %%
def test_centroid_classifer():
    train_points = np.array([[0.], [1.], [5.], [4.], [4.], [4.]])
    train_labels = np.array([0, 0, 0, 1, 1, 1])

    test_points = np.array([[0.], [1.], [2.], [3.], [4.], [5.], [0.]])
    test_labels = np.array([0, 0, 0, 0, 1, 1, 1])

    k = 2
    d = 1

    clf = NearestCentroidClassifier(k, d)
    clf.fit(train_points, train_labels)
    predictions = clf.predict(test_points)
    score = clf.score(test_points, test_labels)
    print(f'Your classifier predicted {predictions}')
    print(f'This gives it a score of {score}')


if __name__ == '__main__':
    test_centroid_classifer()
