```python
python script_final.py 

# outputs predictions.csv
# data_train.pkl and data_test.pkl should be kept in the same directory.

```

### Team Members:
1. Akshay Singh Rana
2. Himanshu Arora


## Final Submission
We have tried and tested with multiple approaches ranging from basic machine learning algorithms to Bidrectional LSTMs. We have mentioned everything in the report, but as stated, we have shared the code only for our best score.

We are using a Tfidf Vectorizer with a Voting Classifier and achieved .60155 score on public set and 0.59238 on private set. We have used three classfiers namely.
1. Complement Naive Bayes with alpha = 1
2. Multinomial Naive Bayes with alpha = 0.25
3. Multi Layered Perceptron with one hidden layer of 250 units.

The above Hyper params have been found after rigorous grid search and multiple tries.

Since we used tfidf vectors, the training might take some time due to tha large parameters the Multi Layer Perceptron has to learn. 

## MileStone1
The Class NB contains three methods i.e.
preprocess(): Removing stopwords and tokenization.
fit(): Create a vocabulary of all the words in training set.
predict(): return the prediction after calculating using naive bayes. 
