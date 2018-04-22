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
from operator import itemgetter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

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
        
        try:
            b = (type(X_train) == np.ndarray)
        except ValueError:
            print "X_train is not numpy.ndarray"
            return False

        
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
            distance = 0
        return distance_array

    
    
    def _findNeighbors(self, x):
        '''
        bussines = 0 #0
        film = 0    #1
        football = 0    #2
        politics = 0    #3
        technology = 0  #4
        '''
        mylist = []
        distanceArray = self.euclideanDistance(x)
        mylist = zip(distanceArray, self.y_)
        mylist.sort(key = itemgetter(0) )
        mylist = mylist[0:self.neighborsNumber]
        votes = {}
        for i in range(len(mylist)):
            vote = mylist[i][1]
            if vote in votes:
                votes[vote] +=1
            else:
                votes[vote] = 1
        return max(votes.iteritems(), key = itemgetter(1))[0]
    
    
    def predict(self, X, y=None):
        
        check_is_fitted(self, ['X_', 'y_'])  
        X = check_array(X)
        if not isinstance(X,np.ndarray):
            print X
            print type(X)
            raise ValueError ('malformed data in predict')
        else:
            X = check_array(X)
                      
        

        #voting

        
        
        
        return np.asarray([self._findNeighbors(x) for x in X])