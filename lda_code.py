import sys
import csv
import string
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import scipy
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim.models import LdaModel,TfidfModel
import random

csv.field_size_limit(sys.maxsize)

stop = stopwords.words('english')

def parse_tokenize_stem(texts):
    r_texts = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tokens = [i for i in tokens if i not in stop]
        stems = []
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        for item in tokens:
            stems.append(stemmer.stem(item))
        r_texts.append(stems)
    return r_texts

def topic_modelling(filename):
    docs = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter = '\t')
        for row in spamreader:
            docs.append(row[3].lower().translate(None, string.punctuation))
    texts = parse_tokenize_stem(docs)
    pickle.dump(texts, open('stemmed_text', 'w'))
    #stoplist = set('for a of the and to in'.split())
    #texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in docs]
    #print type(texts)
    #return
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print 'corpus done'
    print 'starting model'
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, 
                    num_topics=25, update_every=1, chunksize=10000, passes=1)
    val = lda.print_topics(num_topics=25, num_words=10)
    lda.save('lda_25.gensim')
    print val

if __name__ == "__main__":
    topic_modelling("train_data_valid.csv")
