import sys
import csv
import string
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

csv.field_size_limit(sys.maxsize)

# (index, title, identifier, html, tag) is a row in our dataset
# (index, features, tag) is the expectation

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems

def extract_features(filename):
	token_dict = {}
	with open(filename, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = '\t')
		for row in spamreader:
			token_dict[row[0]] = row[3].lower().translate(None, string.punctuation)
	tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 10000, sublinear_tf = True)
	term_document_matrix = tfidf.fit_transform(token_dict.values())
	with open('term_doc_mtx_10000', 'w') as f:
		pickle.dump(term_document_matrix, f)
	with open('mapping_10000', 'w') as f:
		pickle.dump(tfidf.get_feature_names(), f)


if __name__ == "__main__":
	extract_features("train_data_valid.csv")
