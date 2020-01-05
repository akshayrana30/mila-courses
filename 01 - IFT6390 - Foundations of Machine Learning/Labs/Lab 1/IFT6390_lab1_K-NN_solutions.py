import numpy as np


# Ex 1
def minkowski_vec(x1, x2, p=2.0):
    dist = (np.abs(x1 - x2)**p).sum()**(1.0/p)
    return dist

# for testing
a = np.ones((4,))
b = np.zeros((4,))
print(minkowski_vec(a, b))
print(minkowski_vec(a, a))


# Ex 2
def minkowski_mat(x1, X2, p=2.0):
    dist = (np.abs(x1 - X2)**p).sum(axis=1)**(1.0/p)
    return dist

# solution with np.tile and np.newaxis:
#def minkowski_mat(x1, X2, p=2.0):
#    dist = (np.abs(np.tile(x1[np.newaxis, :], (len(X2), 1)) - X2)**p).sum(axis=1)**(1.0/p)
#    return dist

# for testing
a = np.ones((4,))
b = np.random.randint(5, size=(10,4))
print(minkowski_mat(a,b))


# Ex 3
def knn(x, data, p=2):
    feats = data[:,:-1]
    targets = data[:,-1]
    dist = minkowski_mat(x, feats, p)
    return targets[np.argmin(dist)]

######## Conclusion #########
iris = np.loadtxt('iris.txt')

predictions = np.zeros(iris.shape[0])
for i in range(iris.shape[0]):
    predictions[i] = knn(iris[i,:-1],iris)

targets = iris[:,-1]
print("error rate:",(1.0-(predictions==targets).mean())*100.0)

######## Bonus #########
indexes = np.arange(iris.shape[0])
# set the random seed so we have exact reproducibility
np.random.seed(3395)
np.random.shuffle(indexes)

train_set = iris[indexes[:50]]
test_set = iris[indexes[50:]]

# predictions on the training set
train_predictions = np.zeros(train_set.shape[0])
for i in range(train_set.shape[0]):
    train_predictions[i] = knn(train_set[i,:-1],train_set)

# predictions on the testing set
test_predictions = np.zeros(test_set.shape[0])
for i in range(test_set.shape[0]):
    test_predictions[i] = knn(test_set[i,:-1],train_set)

print("Training data error rate", (1.0-(train_predictions==train_set[:,-1]).mean())*100.0)
print("Testing data error rate", (1.0-(test_predictions==test_set[:,-1]).mean())*100.0)

