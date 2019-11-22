import numpy as np
import pandas as pd
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


def milestone1():
    # Milestone1 code for self implementing Naive Bayes with Bag of Words approach.
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



from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":

    print("Loading data..")
    data_train = np.load("./data_train.pkl", allow_pickle = True)
    data_test = np.load("./data_test.pkl", allow_pickle = True)
    print("Loaded train and test data..")

    print("Converting texts to vectors..")
    tfidf = TfidfVectorizer( stop_words = "english", sublinear_tf = True, strip_accents='unicode')
    tfidf.fit(data_train[0])

    X_train = tfidf.transform(data_train[0])
    X_test = tfidf.transform(data_test)

    print("Converted texts to tfidf vectors..")

    print("Initializing all the classifiers")
    clf_cb = ComplementNB(alpha = 1)
    clf_mnb = MultinomialNB(alpha = 0.25)
    clf_mlp = MLPClassifier(hidden_layer_sizes = (250,), max_iter=1, batch_size=64)

    from sklearn.ensemble import VotingClassifier
    clf_vot = VotingClassifier(estimators=[
                                ('1',clf_mnb),
                                ('2',clf_cb),
                                ('3', clf_mlp)], voting='soft')
    
    print("Loaded all the models in Voting Classifier")
    print("Training started..")
    clf_vot.fit(X_train,data_train[1])
    print("Training finished..")
    
    print("Predicting test data and writing in prediction.csv")
    y_pred = clf_vot.predict(X_test)

    df = pd.DataFrame({"Category":y_pred})
    df.to_csv("./prediction.csv", index=True, index_label = "Id", columns=["Category"])





    


