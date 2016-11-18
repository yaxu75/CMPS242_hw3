from math import log, exp
from NaiveBayse_with_smooth import LNaiveBayesClassifier
from PreProcessing import trainfeats
from scipy.optimize import minimize

Lclassifier = LNaiveBayesClassifier.train(trainfeats)
WordList = Lclassifier.most_informative_features(n=10)

x0= [0] * len(WordList)

fun = lambda theta: f(theta)

def f(theta):
    cost = 0
    for featuresets, label in trainfeats:
        text = [None]
        for feature in featuresets:
            text.append(feature)
        xi = [None] * len(WordList)
        kn = 0
        i = 0
        if label == 'ham':
            y = 0
        if label == 'spam':
            y = 1

        for word, value in WordList:
            if word in text:
                xi[i] = 1
                m = xi[i] * theta[i]
                kn += m
                i += 1
            else:
                xi[i] = 0
                i += 1

        cost += log(1 + exp(kn)) - y*kn
    return cost



res = minimize(fun, x0, method='Nelder-Mead')
print res.x

[  9.0811247   -4.84882243  -5.99124626   9.5272596   -5.66214355
  -5.29375258  -4.13511453  -3.47180958  13.52139869  -7.07628822]