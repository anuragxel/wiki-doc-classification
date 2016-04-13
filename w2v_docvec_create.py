import numpy as np
import csv
import string
import nltk
import sys
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

csv.field_size_limit(sys.maxsize)

#this part reads the word vectors and sepates 
with open('glove.6B.50d.txt') as f:
    content = f.readlines()
m = []
for i in content:
    t = i.split()
    if(len(t))!=51:
        print 'oops'
    else:
        m.append(t)
m = np.asarray(m)
words,vecs = m[:,0],m[:,1:].astype(float)
print words.shape
print vecs.shape
del m
docvecs = []
stops = stopwords.words('english')
with open('dataset.csv') as csvfile:
    spamreader = csv.reader(csvfile,delimiter='\t')
    for row in spamreader:
        text=row[3].lower().translate(None,string.punctuation)
        tokens = nltk.word_tokenize(text)
        tokens = [i for i in tokens if i not in stops]
        k = len(tokens)
        vec = np.zeros(vecs.shape[1])
        for word in tokens:
            try:
                vec += vecs[np.argwhere(words==word)[0][0]]
            except:
                pass
        docvecs.append(vec/k)
print np.asarray(docvecs).shape
with open('w2vdocvecs','w') as g:
    pickle.dump(docvecs,g)














































