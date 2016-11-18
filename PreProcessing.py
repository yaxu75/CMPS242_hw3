import os
import codecs
from nltk import tokenize

import sys
reload(sys)
sys.setdefaultencoding('latin-1')

def bagofwords(words):
    return dict([(word, True) for word in words])

label_1 = "ham"
label_2 = "spam"
ham_list=[]
spam_list=[]

root = '.../data'
directories = ['enron1','enron2','enron3','enron4','enron5']
labels=['ham','spam']

ham_features=[]
spam_features=[]
for directory in directories:
    label = labels[0]
    name = os.path.join(root,directory,label)
    for file in os.listdir(name):
        fullname = os.path.join(name,file)
        buffer = codecs.open(fullname,'r').read()
        ham_features.append((bagofwords(tokenize.word_tokenize(buffer)), label))

    label = labels[1]
    name = os.path.join(root,directory,label)
    for file in os.listdir(name):
        fullname = os.path.join(name,file)
        buffer = codecs.open(fullname,'r').read()
        spam_features.append((bagofwords(tokenize.word_tokenize(buffer)), label))

trainfeats = ham_features[:len(ham_features)] + spam_features[:len(spam_features)]

directories = ['enron6']
tham_features=[]
tspam_features=[]

for directory in directories:
    label = labels[0] # ham
    name = os.path.join(root,directory,label)
    for file in os.listdir(name):
        fullname = os.path.join(name,file)
        buffer = codecs.open(fullname,'r').read()
        tham_features.append((bagofwords(tokenize.word_tokenize(buffer)), label))

    label = labels[1]
    name = os.path.join(root,directory,label)
    for file in os.listdir(name):
        fullname = os.path.join(name,file)
        buffer = codecs.open(fullname,'r').read()
        tspam_features.append((bagofwords(tokenize.word_tokenize(buffer)), label))

testfeats = tham_features[:len(tham_features)] + tspam_features[:len(tspam_features)]




