from __future__ import division
from math import exp, log
from PreProcessing import testfeats,trainfeats
from NaiveBayse_with_smooth import LNaiveBayesClassifier


Lclassifier = LNaiveBayesClassifier.train(trainfeats)
WordList = Lclassifier.most_informative_features(n=10)
weight = [9.0811247, -4.84882243, -5.99124626, 9.5272596, -5.66214355,
          -5.29375258, -4.13511453,-3.47180958, 13.52139869, -7.07628822]

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
            m = xi[i] * weight[i]
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


print hit4

accuracy_spam = hit1/(hit1 +hit4)
accuracy_ham = hit2/(hit2 + hit3)

