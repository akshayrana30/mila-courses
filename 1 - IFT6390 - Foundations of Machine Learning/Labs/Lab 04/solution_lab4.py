import sys

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)
np.random.seed(2)


##############################################################################
#
# QUESTION 1
#
##############################################################################

def error_counter(z):
    return (z < 0).astype(int)


def linearloss(z):
    return .5 * (z - 1) ** 2


def logisticloss(z):
    return np.log(1 + np.exp(-z))


def perceptronloss(z):
    return np.maximum(0, -z)


def svmloss(z):
    return np.maximum(0, 1 - z)


zz = np.linspace(-3, 3, 1000)
plt.figure()
for loss in [error_counter, linearloss, logisticloss, perceptronloss, svmloss]:
    plt.plot(zz, loss(zz), label=loss.__name__)

plt.ylim(-.5, 4.5)
plt.legend()
plt.grid()
plt.show()

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    iris = np.loadtxt('http://www.iro.umontreal.ca/~dift3395/files/iris.txt')
else:
    iris = np.loadtxt('iris.txt')


##############################################################################
#
# PREPROCESSING
#
##############################################################################

def preprocess(data, label_subset, feature_subset, n_train):
    """Randomly split data into a train and test set
    with the subset of classes in label_subset and the subset
    of features in feature_subset.
    """
    # extract only data with class label in label_subset
    data = data[np.isin(data[:, -1], label_subset), :]
    # remap labels to [-1, 1]
    if len(label_subset) != 2:
        raise Warning('We are dealing with binary classification.')
    data[data[:, -1] == label_subset[0], -1] = -1
    data[data[:, -1] == label_subset[1], -1] = 1

    # extract chosen features + labels
    data = data[:, feature_subset + [-1]]
    # insert a column of 1s for the bias
    data = np.insert(data, -1, 1, axis=1)

    # separate into train and test
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]
    trainset = data[train_inds]
    testset = data[test_inds]

    # normalize train set to mean 0 and standard deviation 1 feature-wise
    # apply the same transformation to the test set
    mu = trainset[:, :2].mean(axis=0)
    sigma = trainset[:, :2].std(axis=0)
    trainset[:, :2] = (trainset[:, :2] - mu) / sigma
    testset[:, :2] = (testset[:, :2] - mu) / sigma

    return trainset, testset


trainset, testset = preprocess(iris, label_subset=[1, 2], feature_subset=[2, 3], n_train=75)


##############################################################################
#
# HELPER FUNCTIONS
#
##############################################################################


def scatter(theset, marker='o'):
    d1 = theset[theset[:, -1] > 0]
    d2 = theset[theset[:, -1] < 0]
    plt.scatter(d1[:, 0], d1[:, 1], c='b', marker=marker, label='class 1', alpha=.7)
    plt.scatter(d2[:, 0], d2[:, 1], c='g', marker=marker, label='class 0', alpha=.7)
    plt.xlabel('x_0')
    plt.ylabel('x_1')


def finalize_plot(title):
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()


scatter(trainset, marker='x')
scatter(testset, marker='^')
finalize_plot('train and test data')


def decision_boundary(w):
    # hack to avoid changing the boundaries
    xlim = plt.xlim()
    ylim = plt.ylim()

    xx = np.linspace(-10, 10, 2)
    yy = -(w[2] + w[0] * xx) / w[1]
    plt.plot(xx, yy, c='r', lw=2, label='f(x)=0')

    # hack to avoid changing the boundaries
    plt.xlim(xlim)
    plt.ylim(ylim)


w0 = np.array([1, -1, 1])
scatter(trainset)
decision_boundary(w0)
finalize_plot('A random classifier')


##############################################################################
#
# BASE CLASS
#
##############################################################################


class LinearModel:
    """"Abstract class for all linear models.
    -------
    Classe parent pour tous les modèles linéaires.
    """

    def __init__(self, w0, reg):
        self.w = np.array(w0, dtype=float)
        self.reg = reg

    def predict(self, X):
        return np.dot(X, self.w)

    def test(self, X, y):
        return np.mean(self.predict(X) * y < 0)

    def loss(self, X, y):
        return 0

    def gradient(self, X, y):
        return self.w

    def train(self, data, stepsize, n_steps):
        X = data[:, :-1]
        y = data[:, -1]
        losses = []
        errors = []

        for _ in range(n_steps):
            self.w -= stepsize * self.gradient(X, y)
            losses += [self.loss(X, y)]
            errors += [self.test(X, y)]

        print("Training {} completed: the train error is {:.2f}%".format(self.__class__.__name__, errors[-1] * 100))
        return np.array(losses), np.array(errors)


def test_model(modelclass, w0=[-1, 1, 1], reg=.1, stepsize=.2):
    model = modelclass(w0, reg)
    training_loss, training_error = model.train(trainset, stepsize, 100)
    print("The test error is {:.2f}%".format(
        model.test(testset[:, :-1], testset[:, -1]) * 100))
    print('Final weights: ', model.w)

    # learning curves
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 2))
    ax0.plot(training_loss)
    ax0.set_title('loss')
    ax1.plot(training_error)
    ax1.set_title('error rate')
    fig.suptitle(modelclass.__name__)

    # data plot
    plt.figure()
    scatter(trainset, marker='x')
    scatter(testset, marker='^')
    decision_boundary(model.w)
    finalize_plot(modelclass.__name__)


test_model(LinearModel)


##############################################################################
#
# QUESTION 2
#
##############################################################################

class LinearRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        return .5 * np.mean((self.predict(X) - y) ** 2) + .5 * self.reg * np.sum(self.w ** 2)

    def gradient(self, X, y):
        return ((self.predict(X) - y)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


test_model(LinearRegression)


##############################################################################
#
# QUESTION 3
#
##############################################################################


class Perceptron(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        return .5 * np.mean(np.maximum(0, -y * self.predict(X))) + .5 * self.reg * np.sum(
            self.w ** 2)

    def gradient(self, X, y):
        active = (y * self.predict(X) < 0).astype(float)
        return - ((y * active)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


test_model(Perceptron, reg=0, stepsize=1)


##############################################################################
#
# QUESTION 4
#
##############################################################################


class SVM(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        return np.mean(np.maximum(0, 1 - y * self.predict(X))) + .5 * self.reg * np.sum(
            self.w ** 2)

    def gradient(self, X, y):
        active = (y * self.predict(X) < 1).astype(float)
        return - ((y * active)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


test_model(SVM, reg=.001, stepsize=.5)


##############################################################################
#
# QUESTION 5
#
##############################################################################


class LogisticRegression(LinearModel):

    def __init__(self, w0, reg):
        super().__init__(w0, reg)

    def loss(self, X, y):
        return np.mean(np.log(1 + np.exp(-y * self.predict(X)))) + .5 * self.reg * np.sum(
            self.w ** 2)

    def gradient(self, X, y):
        probas = 1 / (1 + np.exp(y * self.predict(X)))
        return - ((y * probas)[:, np.newaxis] * X).mean(axis=0) + self.reg * self.w


test_model(LogisticRegression)
