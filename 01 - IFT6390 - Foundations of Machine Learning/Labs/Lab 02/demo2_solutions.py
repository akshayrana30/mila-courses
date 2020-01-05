import numpy as np
import random
import matplotlib.pyplot as plt
import time


def minkowski_mat(x, Y, p=2):
    return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)


def conf_matrix(testlabels, predlabels):

    n_classes = int(max(testlabels))
    matrix = np.zeros((n_classes, n_classes))

    for (test, pred) in zip(testlabels, predlabels):
        matrix[int(test - 1), int(pred - 1)] += 1

    return matrix


# function plot
def gridplot(classifier, train, test, n_points=50):
    train_test = np.vstack((train, test))
    (min_x1, max_x1) = (min(train_test[:, 0]) - .25, max(train_test[:, 0]) + .25)
    (min_x2, max_x2) = (min(train_test[:, 1]) - .25, max(train_test[:, 1]) + .25)

    xgrid = np.linspace(min_x1, max_x1, num=n_points)
    ygrid = np.linspace(min_x2, max_x2, num=n_points)

    # calculates the cartesian product between two lists
    # and stores the result in an array
    thegrid = np.array(combine(xgrid, ygrid))

    predictedClasses = classifier.compute_predictions(thegrid)

    # The grid
    plt.pcolormesh(xgrid, ygrid, predictedClasses.reshape((n_points, n_points)).T, cmap=plt.cm.cool, alpha=.1)
    # Training data points
    plt.scatter(train[:, 0], train[:, 1], c=train[:, -1], cmap=plt.cm.cool, marker='v', s=70, label='train')
    # Test data points
    plt.scatter(test[:, 0], test[:, 1], c=test[:, -1], cmap=plt.cm.cool, marker='s', s=70, label='test')

    plt.legend()
    plt.show()


# http://code.activestate.com/recipes/302478/
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


class NeighborhoodClassifier:
    def __init__(self, parzen=False, dist_func=minkowski_mat, k=1, radius=0.4):
        self.parzen = parzen
        self.dist_func = dist_func
        self.k = k
        self.radius = radius

    # The train function for knn / Parzen windows is really only storing the dataset
    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    # The prediction function takes as input test_data and returns an array containing the predicted classes.
    def compute_predictions(self, test_data):
        # Initialization of the count matrix and the predicted classes array
        num_test = test_data.shape[0]
        counts = np.ones((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            # Find the distances to each training set point using dist_func
            distances = self.dist_func(ex, self.train_inputs)

            # Go through the training set to find the neighbors of the current point (ex)
            # You will distinguish between Parzen and KNN here
            ind_neighbors = []
            if self.parzen:
                radius = self.radius
                while len(ind_neighbors) == 0:
                    ind_neighbors = np.array([j for j in range(len(distances)) if distances[j] < radius])
                    radius *= 2
            else:
                ind_neighbors = np.argsort(distances)[:self.k]

            # Calculate the number of neighbors belonging to each class and write them in counts[i,:]
            cl_neighbors = list(self.train_labels[ind_neighbors] - 1)
            for j in range(min(len(cl_neighbors) if self.parzen else self.k, self.train_inputs.shape[0])):
                counts[i, cl_neighbors[j]] += 1

            # From the counts matrix, define classes_pred[i] (don't forget that classes are labeled from 1 to n)
            classes_pred[i] = np.argmax(counts[i, :]) + 1.

        return classes_pred


# load iris
iris = np.loadtxt('iris.txt')
data = iris

# Number of classes
n_classes = 3
# Size of training set
n_train = 100

# The columns (features) on which to train our model
# For gridplot to work, len(train_cols) should be 2
train_cols = [0, 1]
# The index of the column containing the labels
target_ind = [data.shape[1] - 1]

# Comment to have random (non-deterministic) results
random.seed(3395)
# Randomly choose indexes for the train and test dataset
inds = list(range(data.shape[0]))
random.shuffle(inds)
train_inds = inds[:n_train]
test_inds = inds[n_train:]

# Split the data into both sets
train_set = data[train_inds, :]
train_set = train_set[:, train_cols + target_ind]
test_set = data[test_inds, :]
test_set = test_set[:, train_cols + target_ind]

# Separate the test set into inputs and labels
test_inputs = test_set[:, :-1]
test_labels = test_set[:, -1].astype('int32')
train_inputs = train_set[:, :-1]
train_labels = train_set[:, -1].astype('int32')


# Number of neighbors (k) for knn
k = 3
radius = 0.9
print("We will train ", k, "-NN and a Parzen classifier with radius ", radius, " on ", n_train, " training examples")

# We create the classifiers
knn = NeighborhoodClassifier(parzen=False, dist_func=minkowski_mat, k=k)
parzen = NeighborhoodClassifier(parzen=True, dist_func=minkowski_mat, radius=radius)

# We train the models
knn.train(train_inputs, train_labels)
parzen.train(train_inputs, train_labels)

# We get predictions
t1 = time.clock()
classes_pred_knn = knn.compute_predictions(test_inputs)
t2 = time.clock()
print('It took knn ', t2 - t1, ' seconds to get the predictions on ', test_inputs.shape[0],' test set examples')
t1 = time.clock()
classes_pred_parzen = parzen.compute_predictions(test_inputs)
t2 = time.clock()
print('It took Parzen ', t2 - t1, ' seconds to get the predictions on ', test_inputs.shape[0],' test set examples')


def show_results(model, classes_pred):
    # Confusion Matrix
    confmat = conf_matrix(test_labels, classes_pred)
    print('The confusion matrix is:')
    print(confmat)

    # Test error
    sum_preds = np.sum(confmat)
    sum_correct = np.sum(np.diag(confmat))
    print("The test error is ", round(100 * (1.0 - (float(sum_correct) / float(sum_preds))), 2), "%")

    # The grid size will be = grid_size x grid_size
    grid_size = 200

    if len(train_cols) == 2:
        # Decision boundary
        t1 = time.clock()
        gridplot(model, train_set, test_set, n_points=grid_size)
        t2 = time.clock()
        print('It took ', round(t2 - t1, 2), ' seconds to calculate the predictions on', grid_size * grid_size,
              ' points of the grid')
        if model.parzen:
            filename = 'grid_' + '_radius=' + str(model.radius) + '_c1=' + str(train_cols[0]) + '_c2=' + str(
                train_cols[1]) + '.png'
        else:
            filename = 'grid_' + '_k=' + str(model.k) + '_c1=' + str(train_cols[0]) + '_c2=' + str(
                train_cols[1]) + '.png'
        print('We will save the plot into {}'.format(filename))
        plt.savefig(filename, format='png')
    else:
        print('Too many dimensions (', len(train_cols), ') to print the decision boundary')


print('KNN classifier')
show_results(knn, classes_pred_knn)
print('\nPARZEN classifier')
show_results(parzen, classes_pred_parzen)


def get_test_error(k):
    knn = NeighborhoodClassifier(parzen=False, dist_func=minkowski_mat, k=k)

    knn.train(train_inputs, train_labels)

    classes_pred_knn = knn.compute_predictions(test_inputs)

    confmat = conf_matrix(test_labels, classes_pred_knn)

    sum_preds = np.sum(confmat)
    sum_correct = np.sum(np.diag(confmat))

    return 1.0 - sum_correct / sum_preds


plt.plot(range(1, 100), [get_test_error(k) for k in range(1, 100)], label='test error')
plt.legend()
plt.xlabel('number of neighbors')
plt.show()
