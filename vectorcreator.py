import numpy as np


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
print m.shape
print m
words,vecs = m[:,0],m[:,1:-1].astype(float)
print words
print vecs[:1]
