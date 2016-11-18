from __future__ import division
from math import exp, log
from PreProcessing import testfeats,trainfeats
from NaiveBayse_with_smooth import LNaiveBayesClassifier

Lclassifier = LNaiveBayesClassifier.train(trainfeats)
WordList = Lclassifier.most_informative_features(n=10)

wmax = [7.29664355,  -4.23504639,  -2.74439301,   4.68504556,  -0.8859894,
  -7.40287308,   1.07740634,  -4.02572307,  11.77294321, -11.23023709]
sum = 0
hit1 = 0
hit2 = 0
hit3 = 0
hit4 = 0
for featuresets, label in testfeats:
    text = [None]
    for feature in featuresets:
        text.append(feature)
    xi = [None] * len(WordList)
    kn = 0
    i = 0

    for word, value in WordList:
        if word in text:
            xi[i] = 1
            m = xi[i] * wmax[i]
            kn += m
            i += 1
        else:
            xi[i] = 0
            i += 1
        value = exp(kn)/(1+exp(kn))

    if value > 0.5 and label == 'spam':
        hit1 += 1
        sum += 1
    elif value < 0.5 and label == 'ham':
        hit2 += 1
        sum += 1
    elif value > 0.5 and label == 'ham':
        hit3 += 1
        sum += 1
    elif value < 0.5 and label == 'spam':
        hit4 += 1
        sum += 1



baccuracy_spam = hit1/(hit1 +hit4)
baccuracy_ham = hit2/(hit2 + hit3)
print(baccuracy_spam)
print(baccuracy_ham)
