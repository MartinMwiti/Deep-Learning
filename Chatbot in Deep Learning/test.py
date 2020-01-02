import json
import random
#import tensorflow
#import tflearn
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
import nltk
#nltk.download('punkt')
ls = LancasterStemmer

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
   
ps = PorterStemmer() 

with open("D:\Deep Learning Application\Deep-Learning\Chatbot in Deep Learning\json-file\json file\intents.json") as file:
    data = json.load(file)

print(data["intents"])


words = []
labels = []
docs_x = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        # add all the words next to each other(stacking horizontally), makes it one list instead of multiple lists inside a list created by .append
        words.extend(wrds)
        docs_x.append(pattern)
        doc_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])


words = [ps.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

print(words)