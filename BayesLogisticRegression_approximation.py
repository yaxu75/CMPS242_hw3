from __future__ import division
from math import exp, log
from scipy.optimize import minimize
from PreProcessing import testfeats,trainfeats
from NaiveBayse_with_smooth import LNaiveBayesClassifier

Lclassifier = LNaiveBayesClassifier.train(trainfeats)
WordList = Lclassifier.most_informative_features(n=10)

# assume mean of prior w0 to be the frequency of dirctory word in testfeats
w0 = [9.0811247, -4.84882243, -5.99124626, 9.5272596, -5.66214355,
          -5.29375258, -4.13511453,-3.47180958, 13.52139869, -7.07628822]


# assume covariance matrix of prior to be I
# doing Laplace approximation
fun = lambda wmax: f(wmax)

def f(wmax):
    cost = 0
    for featuresets, label in trainfeats:
        t1 = 0

        a = 0
        while a < len(WordList):
            t1 += (wmax[a] - w0[a])**2
            a += 1

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
                m = xi[i] * wmax[i]
                kn += m
                i += 1
            else:
                xi[i] = 0
                i += 1
        t2 = y * log(1/(1+exp(kn))) + (1-y) * log(exp(kn)/(1+exp(kn)))

        cost += 0.5*t1 - t2
    return cost

x0 = [0] * len(WordList)
res = minimize(fun, x0, method='Nelder-Mead')
print res.x

res.x = [ 7.29664355,  -4.23504639,  -2.74439301,   4.68504556,  -0.8859894,
  -7.40287308,   1.07740634,  -4.02572307,  11.77294321, -11.23023709]