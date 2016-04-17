import sys
import csv
import string
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import random
import time

csv.field_size_limit(sys.maxsize)

# (index, title, identifier, html, tag) is a row in our dataset
# (index, features, tag) is the expectation

class LabeledLineSentence(object):
    def __init__(self, csv_file):
       self.csv_file = open(csv_file, 'rb')
       self.spamreader = csv.reader(self.csv_file, delimiter = ',')
    def __iter__(self):
        for row in spamreader:
            doc = row[3].lower().translate(None, string.punctuation)
            yield LabeledSentence(doc.split(), [row[4]])
    def to_array(self):
        self.sentences = []
        new_reader = csv.reader(self.csv_file, delimiter = ',')
        for row in new_reader:
            doc = row[3].lower().translate(None, string.punctuation)
            self.sentences.append(LabeledSentence(doc.split(), [row[4]]))
        return self.sentences
    def sentences_perm(self):
        random.shuffle(self.sentences)
        return self.sentences


def extract_features(filename):
    sentences = LabeledLineSentence(filename)
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    print "starting vocab building"
    model.build_vocab(sentences.to_array())
    print "starting training"
    for epoch in range(10):
        print epoch
        model.train(sentences.sentences_perm())
    model.save('./imdb.d2v')
    print model.most_similar('good')

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems

def load_and_emit_vectors(filename):
    model = Doc2Vec.load('../project_snapshot/imdb.d2v')
    dataset = pickle.load(open('gensim_data.frmt'))
    vecs = []
    i = 0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t')
        for row in spamreader:
            i += 1
            val = tokenize(row[3].lower().translate(None, string.punctuation))
            x = model.infer_vector(val)
            print i
            vecs.append(x)
    pickle.dump(np.array(vecs), open('doc2vec_features', 'w'))

if __name__ == "__main__":
    #extract_features("train_data_valid.csv")
    load_and_emit_vectors("train_data_valid.csv")
