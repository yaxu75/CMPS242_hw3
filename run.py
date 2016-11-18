import nltk
from NaiveBayse_without_smooth import NaiveBayesClassifier
from PreProcessing import trainfeats, testfeats
from NaiveBayse_with_smooth import LNaiveBayesClassifier
from BinaryLogisticRegression_prediction import accuracy_ham, accuracy_spam
from BayesLogisticRegression_prediction import baccuracy_ham, baccuracy_spam

print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print(('Accuracy of Naive Bayes without Laplace smoothing:', nltk.classify.util.accuracy(classifier, testfeats)))
classifier.show_most_informative_features()

Lclassifier = LNaiveBayesClassifier.train(trainfeats)
print(('Accuracy of Naive Bayes Laplace smoothing:', nltk.classify.util.accuracy(Lclassifier, testfeats)))
Lclassifier.show_most_informative_features()

print('Accuracy of Binary Logistic Regression for spam mail:', accuracy_spam)
print('Accuracy of Binary Logistic Regression for ham mail:', accuracy_ham)

print('Accuracy of Bayes Logistic Regression for spam mail:', baccuracy_spam)
print('Accuracy of Bayes Logistic Regression for ham mail:', baccuracy_ham)

