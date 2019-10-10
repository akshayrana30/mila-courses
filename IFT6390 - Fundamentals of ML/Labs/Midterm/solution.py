import numpy as np

# %%
'''# Plain Python'''


# %%
def ones_at_the_end(x):
    no_ones = []
    ones = []
    for el in x:
        if el == 1:
            ones.append(el)
        else:
            no_ones.append(el)
    no_ones.reverse()
    return no_ones + ones


# %%
def final_position(instructions):
    count = {'right': 0, 'left': 0, 'up': 0, 'down': 0}
    for word in instructions.split():
        if word in ['right', 'left', 'up', 'down']:
            count[word] += 1
    x = count['right'] - count['left']
    y = count['up'] - count['down']
    return x, y


# %%
def find_bins(input_list, k):
    sorted_inp = sorted(input_list)
    h = int(len(sorted_inp) / k)
    ans = [sorted_inp[0] - 1.]
    for i in range(k - 1):
        ans.append((sorted_inp[h * (i + 1) - 1] + sorted_inp[h * (i + 1)]) / 2)
    ans.append(sorted_inp[-1] + 1.)
    return ans


# %%
def steps_to_one(i):
    n_steps = 0
    cur = i
    while cur != 1:
        n_steps += 1
        if cur % 2 == 0:
            cur = cur / 2
        else:
            cur = 3 * cur + 1
    return n_steps


# %%
'''# Numpy'''


# %%
def even_odd_ordered(X):
    even_idx = (X % 2) == 0
    odd_idx = (X % 2) != 0
    Z = np.concatenate((X[even_idx], X[odd_idx]))
    return Z


# %%
def data_normalization(X):
    Z = X[:, :-1]
    mu = np.mean(Z, axis=0, keepdims=True)
    std = np.std(Z, axis=0, keepdims=True)
    Z = (Z - mu) / (std + 1e-8)
    Z = np.column_stack((Z, X[:, -1]))
    return Z


# %%
def heavyball_optimizer(x, inputs, alpha=0.9, beta=0.1):
    x_prev = np.zeros_like(x)
    for i in range(len(inputs)):
        grad = inputs[i]
        x_next = x - (alpha * grad) + beta * (x - x_prev)
        x_prev = x
        x = x_next
    return x


# %%
def entropy(p):
    if not np.isclose(np.sum(p), 1):
        return None

    ent = 0
    for i in p:
        if i > 0:
            ent += -1 * (i * np.log2(i))
        elif i == 0:
            ent += 0
        else:
            return None
    return ent


# %%
'''# Machine Learning'''


# %%
class NearestCentroidClassifier:
    def __init__(self, k, d):
        """Initialize a classifier with k classes in dimension d."""
        self.k = k
        self.centroids = np.zeros((k, d))

    def fit(self, X, y):  # question A
        """Compute the centroid of each class, and store it in self.centroids"""
        for i in range(self.k):
            self.centroids[i] = np.mean(X[y == i], axis=0)

    def predict(self, X):  # question B
        """Return the predicted class for each row in matrix X"""
        distances = np.sum((X[:, np.newaxis] - self.centroids[np.newaxis, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)

    def score(self, X, y):  # question C
        """Return the accuracy of self on points in X with labels y"""
        return np.mean(self.predict(X) == y)
