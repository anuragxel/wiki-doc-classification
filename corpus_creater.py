import numpy as np
import csv
import sys
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

csv.field_size_limit(sys.maxsize)

stops = stopwords.words('english')
def preprocess(line):
    tokens = nltk.word_tokenize(line)
    tokens = [i for i in tokens if i not in stops if len(i)>2]
    return ' '.join(tokens)

''' 
    this file takes dataset.csv file
    redirect the output into a corpus.txt file
    this corpus.txt meets the specifications given in github "SentenceRepresentation" rpository
    https://github.com/fh295/SentenceRepresentation
'''
with open('dataset.csv') as csvfile:
    spamreader = csv.reader(csvfile,delimiter='\t')
    for row in spamreader:
        text = row[3].lower()
        lines = text.split('.')
        for line in lines:
             modfline = preprocess(line)
             if(len(modfline)>15):
                 print modfline.translate(None,string.punctuation)
