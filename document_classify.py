import numpy as np
import sklearn
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from operator import itemgetter
import csv
from sklearn.lda import LDA
import nltk
import random
import pickle
import sys

csv.field_size_limit(sys.maxsize)

class meta_linear(object):
    def __init__(self, learner):
        self.L = learner

    def __str__(self):
        return self.L.__str__()

    def train(self,X_train,Y_train):
        self.L.fit(X_train,Y_train)

    def predict(self,X_test):
        return self.L.predict(X_test)

class multinomial_bayes(meta_linear):
    def __init__(self):
        super(self.__class__, self).__init__(MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True))

class SGD_SVM(meta_linear):
    def __init__(self):
        super(self.__class__, self).__init__(SGDClassifier( alpha=0.0003, average=False, class_weight=None,
                                                            eta0=0.0,   l1_ratio=0.15,
                                                            learning_rate='optimal', loss='hinge', n_jobs=1,
                                                            penalty='l2', warm_start=False))


class k_fold_cross_validation(object):
    '''
        The class will take an statistical class and training set and parameter k.
        The set will be divided wrt to k and cross validated using the statistical
        model provided.
        The statistical class should have two methods and no constructor args -
        method train(training_x, training_y)
        method predict(x_test_val)
    '''
    def __init__(self,k,stat_class,x_train,y_train):
        self.k_cross = float(k)
        self.stat_class = stat_class
        self.x_train = x_train
        self.y_train = y_train
        self.values = []

    def hit_rate(self, gold, pred):
        pred =np.array(pred)
        pred = pred[0].T
        if gold.shape != pred.shape:
            raise BaseException()
        acc = 0
        for i in xrange(0,len(gold)):

            if gold[i] == pred[i]:
                acc += 1
        print str(acc)+" are correct out of "+str(len(gold))
        return float(acc)/float(len(gold))

    def execute(self):
        kf = KFold(self.x_train.shape[0], n_folds=self.k_cross)
        own_kappa = []
        for train_idx, test_idx in kf:
            x_train, x_test = self.x_train[train_idx], self.x_train[test_idx]
            y_train, y_test = self.y_train[train_idx], self.y_train[test_idx]
            #dim_red = LDA()
            #x_train = dim_red.fit_transform(x_train, y_train)
            #x_test = dim_red.transform(x_test)
            stat_obj = self.stat_class() # reflection bitches
            stat_obj.train(x_train,y_train)
            y_pred = np.matrix(stat_obj.predict(x_test))
            print y_pred
            accuracy = self.hit_rate(y_test, y_pred) # function comes here 
            self.values.append(accuracy)
        return str(sum(self.values)/self.k_cross)

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def estimate_params(clf, X, y, params):
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=params,
                                   n_iter=n_iter_search)
    random_search.fit(X, y)
    report(random_search.grid_scores_)

if __name__ == "__main__":
    term_document_matrix=pickle.load(open('doc2vec_features')) # loads term_document_matrix
    print term_document_matrix.shape
    labels = []
    cross_valid_k = 5
    with open('train_data_valid.csv','r') as in_file:
        spamreader = csv.reader(in_file, delimiter = '\t')
        for row in spamreader:
            labels.append(row[4])
    labels = np.asarray(labels)
    #pca = PCA(n_components=50)
    #selection = SelectKBest(k=50)
    #combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
    #X_features = combined_features.fit(term_document_matrix.toarray(), labels).transform(term_document_matrix.toarray())
    #print X_features.shape
    #print "NaiveBayes"
    #estimate_params(MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True),term_document_matrix,labels, params ={
    #    "alpha" :  np.arange(0.01,1,0.01)
    #})
    print "SGD_full" 
    estimate_params(SGDClassifier( alpha=0.0003, average=False, class_weight=None,
                                                            eta0=0.0,   l1_ratio=0.15,
                                                            learning_rate='optimal', loss='hinge', n_jobs=1,
                                                            penalty='l2', warm_start=False),term_document_matrix,labels, params ={
        "alpha" :  np.arange(0.01,1,0.01),
        "l1_ratio" : np.arange(0.01,1,0.01),
        "loss" : ["hinge", "log", "huber", "squared_hinge"]
    })
    

    #bayes_k_cross = k_fold_cross_validation(cross_valid_k,multinomial_bayes,term_document_matrix,labels)
    #print "bayes multinomial : " + bayes_k_cross.execute()
    #SGD_k_cross = k_fold_cross_validation(cross_valid_k,SGD_SVM,term_document_matrix,labels)
    #print "stochastic gradient descent : " + SGD_k_cross.execute()
