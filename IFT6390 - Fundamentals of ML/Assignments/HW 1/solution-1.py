import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        return np.mean(iris, axis = 0)[0:4]

    def covariance_matrix(self, iris):
        X = iris[:,:4]
        # mean_X = np.mean(X, axis = 0)
        # len_X = X.shape[0]
        # norm_X = X - mean_X
        # return norm_X.T.dot(norm_X)/len_X
        return np.cov(X, rowvar=False)


    def feature_means_class_1(self, iris):
        iris_class_1 = iris[iris[:,4] == 1]
        return self.feature_means(iris_class_1)

    def covariance_matrix_class_1(self, iris):
        iris_class_1 = iris[iris[:,4] == 1]
        return self.covariance_matrix(iris_class_1)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))

    def euclidean_dist(self, x, Y):
        return np.linalg.norm(x - Y, axis=1)  

    def compute_predictions(self, test_data):
        counts = np.ones((test_data.shape[0], self.n_classes))
        classes_pred = np.zeros(test_data.shape[0])
        for (i, ex) in enumerate(test_data):
            dist = self.euclidean_dist(ex, self.train_inputs)
            neighbours = np.where(dist <= self.h)
            if len(neighbours) == 0:
                classes_pred[i] = draw_rand_label(ex, self.label_list)
                continue

            for label in self.train_labels[neighbours]:
                counts[i,int(label-1)] += 1
            classes_pred[i] = np.argmax(counts[i]) + 1
        return classes_pred



class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)
        self.n_classes = len(np.unique(train_labels))
        y = self.train_labels.astype(int)
        new_y = y.reshape(y.shape[0])
        n_values = np.max(new_y)
        self.one_hot = np.eye(n_values)[new_y-1]

    def parsen_window(self,ex, d, train):
        part1 = 1 / (  ((2*np.pi)**(d/2)) * ((self.sigma)**d))
        part2 = (-1/2) * ( (self.euclidean_dist(ex, train)/self.sigma) ** 2 )        
        final = part1*np.exp(part2)
        return final

    def euclidean_dist(self, x, Y):
        return np.linalg.norm(x - Y, axis=0) 
    
    def compute_predictions(self, test_data):
        n, d = test_data.shape
        counts = np.ones((test_data.shape[0], self.n_classes))
        classes_pred = np.zeros(test_data.shape[0])
        for i,ex in enumerate(test_data):
            prob_j = 0
            for j,train in enumerate(self.train_inputs):
                prob_j += self.parsen_window(ex, d, train)*self.one_hot[j]
            counts[i] = prob_j
            classes_pred[i] = np.argmax(counts[i])+1
        return classes_pred


def split_dataset(iris):
    train = iris[[i for i in range(iris.shape[0]) if i%5 in [0,1,2]]]
    val = iris[[i for i in range(iris.shape[0]) if i%5 in [3]]]
    test = iris[[i for i in range(iris.shape[0]) if i%5 in [4]]]
    return (train, val, test)


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def conf_matrix(self, testlabels, predlabels):
        n_classes = int(max(testlabels))
        matrix = np.zeros((n_classes,n_classes))

        for (test, pred) in zip(testlabels, predlabels):
            matrix[int(test-1), int(pred-1)] += 1
        return matrix
    
    def get_error(self, conf_matrix):
        sum_preds = np.sum(conf_matrix)
        sum_correct = np.sum(np.diag(conf_matrix))
        return 1.0 - (float(sum_correct) / float(sum_preds))


    def hard_parzen(self, h):
        cls = HardParzen(h)
        cls.train(self.x_train, self.y_train)
        y_pred = cls.compute_predictions(self.x_val)
        conf_matrix = self.conf_matrix(self.y_val, y_pred)
        return self.get_error(conf_matrix)

    def soft_parzen(self, sigma):
        cls = SoftRBFParzen(sigma)
        cls.train(self.x_train, self.y_train)
        y_pred = cls.compute_predictions(self.x_val)
        conf_matrix = self.conf_matrix(self.y_val, y_pred)
        return self.get_error(conf_matrix)


def get_test_errors(iris):
    (train, val, test) = split_dataset(iris)
    err = ErrorRate(train[:,0:4], train[:,4:5], test[:,0:4], test[:,4:5])
    error_hard =  err.hard_parzen(1.0)
    error_soft =  err.soft_parzen(0.1)
    return np.array([error_hard, error_soft])


def random_projections(X, A):
    y = np.zeros([X.shape[0],2])
    for i,x in enumerate(X):
        y[i] = np.sqrt(1/2)*np.dot(A.T, x)
    return y


## QUESTIONS NOT GRADED FOR GRADESCOPE, BUT USED FOR REPORT ##

def get_projection(X,A):
    b = np.sqrt(1/2)*np.dot(X, A)
    return b

def rndm_prjction_500():
    (train, val, test) = split_dataset(iris)
    train_labels = train[:,4:5]
    val_labels = val[:,4:5]
    train_inputs = train[:,0:4]
    val_inputs = val[:,0:4]
    h = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    error_matrix_hard = np.zeros([500,9])
    error_matrix_soft = np.zeros([500,9])
    for i in range(500):
        A = np.random.normal(0,1, (4,8))
        train_inputs = get_projection(train[:,0:4], A)
        val_inputs = get_projection(val[:,0:4], A)
        err = ErrorRate(train_inputs, train_labels, val_inputs, val_labels)
        for j,value in enumerate(h):
            error_matrix_hard[i][j] = err.hard_parzen(value)
            error_matrix_soft[i][j] = err.soft_parzen(value)

    hard = np.mean(error_matrix_hard, axis=0)
    soft = np.mean(error_matrix_soft, axis=0)

    plt.figure()
    plt.errorbar(h, hard, yerr=0.2)
    plt.errorbar(h, soft, yerr=0.2)