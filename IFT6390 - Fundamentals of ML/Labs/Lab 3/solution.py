import numpy as np
import matplotlib.pyplot as plt


class GaussianMaxLikelihood:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.mu = np.zeros(n_dims)
        self.sigma_sq = 1.0

    # For a training set, the function should compute the ML estimator of the mean and the variance
    def train(self, train_data):
        self.mu = np.mean(train_data, axis=0)
        self.sigma_sq = np.sum((train_data - self.mu) ** 2.0) / (self.n_dims * train_data.shape[0])

    # Returns a vector of size nb. of test ex. containing the log probabilities of each test
    # example under the model.
    def loglikelihood(self, test_data):
        c = - self.n_dims * np.log(2 * np.pi) / 2 - self.n_dims * np.log(np.sqrt(self.sigma_sq))
        log_prob = c - np.sum((test_data - self.mu) ** 2.0, axis=1) / (2.0 * self.sigma_sq)
        return log_prob


class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    # Returns a matrix of size nb. of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    def loglikelihood(self, test_data):
        log_pred = np.zeros((test_data.shape[0], self.n_classes))
        for i in range(self.n_classes):
            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(test_data) + np.log(
                self.priors[i])
        return log_pred


iris = np.loadtxt('iris.txt')
iris_train = iris[:, :-1]
iris_labels = iris[:, -1]

iris_train1 = iris_train[0:50, :]
iris_train2 = iris_train[50:100, :]
iris_train3 = iris_train[100:150, :]

model_class1 = GaussianMaxLikelihood(4)
model_class2 = GaussianMaxLikelihood(4)
model_class3 = GaussianMaxLikelihood(4)
model_class1.train(iris_train1)
model_class2.train(iris_train2)
model_class3.train(iris_train3)

model_ml = [model_class1, model_class2, model_class3]
priors = [1. / 3, 1. / 3, 1. / 3]

classifier = BayesClassifier(model_ml, priors)


def get_accuracy(test_inputs, test_labels):
    log_prob = classifier.loglikelihood(test_inputs)
    classes_pred = log_prob.argmax(1) + 1
    return np.mean(classes_pred == test_labels)


if __name__ == '__main__':
    print(
        "The training accuracy is : {:.1f} % ".format(100 * get_accuracy(iris_train, iris_labels)))


##############################################################################
#
# ANSWER TO THE BONUS QUESTION
# (MAKE SURE YOU DELETE THIS PART BEFORE YOU MAKE A SUBMISSION ON GRADESCOPE)
#
##############################################################################

class GaussianMaxLikelihood:
    def __init__(self, n_dims, cov_type="isotropic"):
        self.n_dims = n_dims
        self.mu = np.zeros((1, n_dims))
        self.cov_type = cov_type

        if cov_type == "isotropic":
            self.sigma_sq = 1.0
        elif cov_type == "diagonal":
            self.sigma_sq = np.ones(n_dims)
        if cov_type == "full":
            self.cov = np.ones((n_dims, n_dims))

    # For a training set, the function should compute the ML estimator of the mean and the
    # covariance matrix
    def train(self, train_data):
        self.mu = np.mean(train_data, axis=0)
        if self.cov_type == "isotropic":
            self.sigma_sq = np.sum((train_data - self.mu) ** 2.0) / (
                        self.n_dims * train_data.shape[0])
        elif self.cov_type == "diagonale":
            self.sigma_sq = np.sum((train_data - self.mu) ** 2.0, axis=0) / train_data.shape[0]
        if self.cov_type == "full":
            self.cov = np.cov(np.transpose(train_data))

    # Returns a vector of size nb. of test ex. containing the log
    # probabilities of each test example under the model.
    def loglikelihood(self, test_data):

        if self.cov_type == "isotropic":
            # the following line calculates log(normalization constant)
            c = -self.n_dims * np.log(2 * np.pi) / 2 - self.n_dims * np.log(np.sqrt(self.sigma_sq))
            # it is necessary to calculate the value of the log-probability of each test example
            # under the model determined by mu and sigma_sq
            # the vector of probabilities is / will be log_prob
            log_prob = c - np.sum((test_data - self.mu) ** 2.0, axis=1) / (2.0 * self.sigma_sq)
        elif self.cov_type == "diagonal":
            # we take the product of the vector representing the diagonal (np.prod(self.sigma)
            c = -self.n_dims * np.log(2 * np.pi) / 2.0 - np.log(np.prod(self.sigma_sq)) / 2.0
            # we do a summation along axis 1 after having divided by sigma because the latte is
            # also
            # of dimension d
            log_prob = c - np.sum((test_data - self.mu) ** 2.0 / (2.0 * self.sigma_sq), axis=1)
        elif self.cov_type == "full":
            c = -self.n_dims * np.log(2.0 * np.pi) / 2.0
            det = np.linalg.det(self.cov)
            c -= np.log(det) / 2.0

            dmu = test_data - self.mu
            inv = np.linalg.inv(self.cov)

            dxs = np.dot(dmu, inv)
            dxsx = np.sum(dxs * dmu, axis=1)
            log_prob = c - dxsx / 2
        return log_prob


class BayesClassifier:

    def __init__(self, models_ml, priors):
        self.models_ml = models_ml
        self.priors = priors
        if len(self.models_ml) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.models_ml)

    # Returns a matrix of size nb. of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    def loglikelihood(self, test_data, eval_by_group=False):

        log_pred = np.empty((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Here we will have to use modeles_mv [i] and priors to fill in
            # each column of log_pred (it's more efficient to do a entire column at a time)

            log_pred[:, i] = self.models_ml[i].loglikelihood(test_data) + np.log(self.priors[i])

        return log_pred


##############################################################################
#
# PERFORM A (2/3, 1/3) RANDOM SPLIT OF DATA
#
##############################################################################

# Here we provide an example where we do not divide the data into train and test set.
iris = np.loadtxt('iris.txt')
np.random.seed(123)

indices1 = np.arange(0, 50)
indices2 = np.arange(50, 100)
indices3 = np.arange(100, 150)

np.random.shuffle(indices1)
np.random.shuffle(indices2)
np.random.shuffle(indices3)

iris_train1 = iris[indices1[:33]]
iris_test1 = iris[indices1[33:]]
iris_train2 = iris[indices2[:33]]
iris_test2 = iris[indices2[33:]]
iris_train3 = iris[indices3[:33]]
iris_test3 = iris[indices3[33:]]

iris_train = np.concatenate([iris_train1, iris_train2, iris_train3])
iris_test = np.concatenate([iris_test1, iris_test2, iris_test3])

##############################################################################
#
# EXPERIMENTS WITH 4 FEATURES
#
##############################################################################

train_cols = [0, 1, 2, 3]

for cov_type in ['isotropic', 'diagonal', 'full']:
    # We create a model per class (using maximum likelihood)
    model_class1 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class2 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class3 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class1.train(iris_train1[:, train_cols])
    model_class2.train(iris_train2[:, train_cols])
    model_class3.train(iris_train3[:, train_cols])

    # We create a list of all our models
    # We do the same thing for the priors
    # Here the priors are calculated exactly because we know the number of representatives per
    # class.
    # Once you have created a train / test set, they need to be calculated exactly
    model_ml = [model_class1, model_class2, model_class3]
    priors = [0.3333, 0.3333, 0.3333]

    # We create our classifier with our list of Gaussian models and our priors
    classifier = BayesClassifier(model_ml, priors)

    # we can now calculate the log-probabilities according to our models
    log_prob_train = classifier.loglikelihood(iris_train[:, train_cols])
    log_prob_test = classifier.loglikelihood(iris_test[:, train_cols])

    # it now remains to calculate the maximum per class for the classification
    classesPred_train = log_prob_train.argmax(1) + 1
    classesPred_test = log_prob_test.argmax(1) + 1

    print(cov_type)
    print("Error rate (training) %.2f%%" % (
                (1 - (classesPred_train == iris_train[:, -1]).mean()) * 100.0))
    print(
        "Error rate (test) %.2f%%" % ((1 - (classesPred_test == iris_test[:, -1]).mean()) * 100.0))


##############################################################################
#
# VISUALIZATION FUNCTIONS
#
##############################################################################

## http://code.activestate.com/recipes/302478/
def combine(*seqin):
    '''returns a list of all combinations of argument sequences.
for example: combine((1,2),(3,4)) returns
[[1, 3], [1, 4], [2, 3], [2, 4]]'''

    def rloop(seqin, listout, comb):
        '''recursive looping function'''
        if seqin:  # any more sequences to process?
            for item in seqin[0]:
                newcomb = comb + [item]  # add next item to current comb
                # call rloop w/ rem seqs, newcomb
                rloop(seqin[1:], listout, newcomb)
        else:  # processing last sequence
            listout.append(comb)  # comb finished, add to list

    listout = []  # listout initialization
    rloop(seqin, listout, [])  # start recursive process
    return listout


def gridplot(classifier, train, test, n_points=50):
    train_test = np.vstack((train, test))
    (min_x1, max_x1) = (min(train_test[:, 0]), max(train_test[:, 0]))
    (min_x2, max_x2) = (min(train_test[:, 1]), max(train_test[:, 1]))

    xgrid = np.linspace(min_x1, max_x1, num=n_points)
    ygrid = np.linspace(min_x2, max_x2, num=n_points)

    # calculates the Cartesian product between two lists
    # and puts the results in an array
    thegrid = np.array(combine(xgrid, ygrid))

    the_accounts = classifier.loglikelihood(thegrid)
    classesPred = np.argmax(the_accounts, axis=1) + 1

    # The grid
    # So that the grid is prettier
    # props = dict( alpha=0.3, edgecolors='none' )
    plt.scatter(thegrid[:, 0], thegrid[:, 1], c=classesPred, s=50)
    # The training points
    plt.scatter(train[:, 0], train[:, 1], c=train[:, -1], marker='v', s=150)
    # The test points
    plt.scatter(test[:, 0], test[:, 1], c=test[:, -1], marker='s', s=150)

    ## A little hack, because the functionality is missing at pylab ...
    h1 = plt.plot([min_x1], [min_x2], marker='o', c='w', ms=5)
    h2 = plt.plot([min_x1], [min_x2], marker='v', c='w', ms=5)
    h3 = plt.plot([min_x1], [min_x2], marker='s', c='w', ms=5)
    handles = [h1, h2, h3]
    ## end of the hack

    labels = ['grille', 'train', 'test']
    plt.legend(handles, labels)

    plt.axis('equal')
    plt.show()


##############################################################################
#
# EXPERIMENTS WITH 2 features
#
##############################################################################

train_cols = [2, 3]

for cov_type in ['isotropic', 'diagonal', 'full']:
    # We create a model per class (using maximum likelihood)
    model_class1 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class2 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class3 = GaussianMaxLikelihood(len(train_cols), cov_type=cov_type)
    model_class1.train(iris_train1[:, train_cols])
    model_class2.train(iris_train2[:, train_cols])
    model_class3.train(iris_train3[:, train_cols])

    # We create a list of all our models
    # We do the same thing for the priors
    # Here the priors are calculated exactly because we know the number of representatives per
    # class.
    # Once you have created a train / test set, they need to be calculated exactly
    model_ml = [model_class1, model_class2, model_class3]
    priors = [0.3333, 0.3333, 0.3333]

    # We create our classifier with our list of Gaussian models and our priors
    classifier = BayesClassifier(model_ml, priors)

    # we can now calculate the log-probabilities according to our models
    log_prob_train = classifier.loglikelihood(iris_train[:, train_cols])
    log_prob_test = classifier.loglikelihood(iris_test[:, train_cols])

    # it now remains to calculate the maximum per class for the classification
    classesPred_train = log_prob_train.argmax(1) + 1
    classesPred_test = log_prob_test.argmax(1) + 1

    print(cov_type)
    print("Error rate (training) %.2f%%" % (
                (1 - (classesPred_train == iris_train[:, -1]).mean()) * 100.0))
    print(
        "Error rate (test) %.2f%%" % ((1 - (classesPred_test == iris_test[:, -1]).mean()) * 100.0))

    gridplot(classifier,
             iris_train[:, train_cols + [-1]],
             iris_test[:, train_cols + [-1]],
             n_points=50)
