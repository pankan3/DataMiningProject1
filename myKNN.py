#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:36:40 2018

@author: kalo-pc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:11:36 2018

@author: giann
"""


import math
#import operator 
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neighbors import KNeighborsClassifier
from os import path
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from wordcloud import STOPWORDS 
from gensim.parsing.porter import PorterStemmer
#from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


'''
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
'''

'''
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
'''

class myKNNClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, neighborsNumber):
        self.neighborsNumber = neighborsNumber
    
    
    def get_params(self, deep=True):
        return {"neighborsNumber": self.neighborsNumber}
    
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
    def fit(self, X_train, y):
        
        #check if input is correct
        try:
            int(self.neighborsNumber)
        except ValueError:
            print "Number of neighbors must be possitive integer"
            return False
        '''
        try:
            type(X_train) = numpy.ndarray
        except ValueError:
            print "X_train is not numpy.ndarray"
            return False
        '''
        
        X_train, y = check_X_y(X_train, y)
        self.classes_ = unique_labels(y)

        self.X_ = X_train
        self.y_ = y
        
        return self
    
    
    def euclideanDistance(self, X):
        distance = 0
        distance_array = []
        for i in range(len(self.X_)):
            for x in range(len(self.X_[i])):
                distance += pow((X[x] - self.X_[i][x]), 2)
            distance_array.append(math.sqrt(distance))
        return distance_array

    
    
    def _findNeighbors(self, X):
        '''
        bussines = 0 #0
        film = 0    #1
        football = 0    #2
        politics = 0    #3
        technology = 0  #4
        '''
        mylist = []
        distanceArray = self.euclideanDistance(X)
        mylist = zip(distanceArray, self.y_)
        
        mylist.sort(key = lambda tup: tup[0])
        mylist[0:self.neighborsNumber]
        
        final_list =[0, 0, 0, 0, 0]
        for x in mylist:
            if x[1] == 0:
                final_list[0] += 1
            elif x[1] == 1:
                final_list[1] += 1
            elif x[1] == 2:
                final_list[2] += 1
            elif x[1] == 3:
                final_list[3] += 1
            elif x[1] == 4:
                final_list[4] += 1
        return final_list.index(max(final_list))
    
    
    def predict(self, X):
        
        #check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        list_=[self._findNeighbors(x) for x in X]
        print np.asarray(list_).shape
        print self.y_.shape
        
        
        return np.asarray(list_)
    
    '''
    def score(): 
        return self
    '''
        
'''
train_data = pd.read_csv('train_set.csv' , sep = "\t")
 
train_data = train_data[0:10000]

d = path.dirname(__file__)
p = PorterStemmer()
 
stopwords = set(ENGLISH_STOP_WORDS)
stopwords2 = set(STOPWORDS)
stopwords.union(stopwords2)
stopwords.add("said")
stopwords.add("say")
stopwords.add("says")
stopwords.add("will")
stopwords.add("just")

#Transform Category 
le=preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y_train=le.transform(train_data["Category"])

#Stemming
for i in range(0,10000):
        train_data.loc[[i],['Content']] = p.stem_sentence(train_data.loc[[i],['Content']].to_string(header=False,index=False))
        
#Vectorazation
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train = count_vectorizer.fit_transform(train_data['Content'])

#LSI
svd = TruncatedSVD(n_components=107)
X_train = svd.fit_transform(X_train)

print "X_train" 
print X_train
print "Y_train" 
print y_train

#clf = 
#y_predict = cross_val_predict(clf,X_train,y_train,cv=10)


#print "accuracy:" , metrics.accuracy_score(y_train, y_predict)
#print "f_measure:", metrics.f1_score(y_train, y_predict, average ='macro')
#print "precision:", metrics.precision_score(y_train, y_predict, average = 'macro')
#print "recall:", metrics.recall_score(y_train, y_predict, average = 'macro')
'''