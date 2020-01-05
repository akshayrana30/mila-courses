import numpy as np
import re

class NB():
    
    def __init__(self):
        self.voc = {}
        from nltk import download
        download('stopwords')
        from nltk.corpus import stopwords
        self.stopwords = stopwords.words("english")
        from nltk.stem.porter import PorterStemmer
        self.stemmer = PorterStemmer()
        self.vocab_set = set()
        
    def fit(self, X,Y):
        # Creating a dictionary for the vocabulary
        voc = {}
        self.length_of_training_data = len(X)
        for data, label in zip(X, Y):
            words = self.preprocess(data)
            if label not in voc:
                voc[label] = {"num_of_words":0, "number_of_samples":0}
            voc[label]['number_of_samples'] += 1
            for word in words:
                self.vocab_set.add(word)
                if word in voc[label]:
                    voc[label][word] += 1
                    voc[label]["num_of_words"] += 1
                else:
                    voc[label][word] = 1
                    voc[label]["num_of_words"] += 1
        self.voc = voc
        
    def preprocess(self, sentence):
        # Removing stop words and adding stemming.
        result = []
        pattern = re.compile(r'\b\w\w+\b')
        for word in re.findall(pattern, sentence):
            word = word.lower()
            if word not in self.stopwords:
                result.append(self.stemmer.stem(word))
        return result
        
    def predict(self, X, alpha = 0.25):
        y = []
        voc = self.voc
        classes = list(voc.keys())
        for value in X:
            words = self.preprocess(value)
            output_probs = np.zeros(len(voc.keys()))
            for i,label in enumerate(classes):
                prob = np.log(voc[label]['number_of_samples']/self.length_of_training_data) #prior probability
                denominator = voc[label]['num_of_words']+alpha*(len(self.vocab_set))
                for word in words:
                    if word not in voc[label]:
                        prob += np.log((0 + alpha)/denominator)
                    else:
                        prob += np.log((voc[label][word] + alpha)/denominator)
                output_probs[i] = prob
            y.append(classes[np.argmax(output_probs)])
        return y
    
    def accuracy(self, y_test, y_pred):
        return round(np.mean(np.array(y_test) == np.array(y_pred))*100, 2)

if __name__ == "__main__":
    
    # loading data
    try:
        data_train = np.load("./data_train.pkl", allow_pickle = True)
        data_test = np.load("./data_test.pkl", allow_pickle = True)
        print("Data Loaded Successfully")
    except Exception as e:
        print(e)
        print("Please change the paths in the code for loading data")

    # initializing the classifier and training
    try:
        cls_nb = NB()
        print("Naive Bayes initialized. Training started..")
        cls_nb.fit(data_train[0], data_train[1])
        print("Training Successfull")
    except Exception as e:
        print(e)
        print("Unable to train. Please check the error")

    # Predicting the test data with alpha = 0.25 for Laplacian Smoothing. Use alpha = 0 for NB without smoothing.
    try:
        print("Testing started..")
        y_pred = cls_nb.predict(data_test, alpha = 0.25)
        print("Testing Successfull")
    except Exception as e:
        print(e)
        print("Unable to test. Please check the error")
    y_test = "" # if you have samples for testing..load them here
    # You can compute the accuracy below by uncommenting it. 
    # accuracy = cls_nb.predict(y_pred, y_test)
    # print(accuracy)

    # Storing the prediction in csv file.
    try:
        f = open("prediction.csv","w+")
        f.write("Id,Category\n")
        for d,s in enumerate(y_pred):
            f.write("%d,%s\n" %(d,s))
        print("All the prediction are printed in prediction.csv")
    except Exception as e:
        print(e)
        print("Unable to write the predictions")


