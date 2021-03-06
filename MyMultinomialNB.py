#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 17:36:56 2018

@author: kalo-pc
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
 
This is a temporary script file.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_predict
import pandas as pd
from wordcloud import STOPWORDS 
from gensim.parsing.porter import PorterStemmer

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
'''
#Stemming
for i in range(0,10000):
        train_data.loc[[i],['Content']] = p.stem_sentence(train_data.loc[[i],['Content']].to_string(header=False,index=False))
   '''     
#Vectorazation
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train = count_vectorizer.fit_transform(train_data['Content'])
'''
#LSI
svd = TruncatedSVD(n_components=107)
X_train = svd.fit_transform(X_train)
'''

clf = MultinomialNB()
y_predict = cross_val_predict(clf,X_train,y_train,cv=10)


print "accuracy:" , metrics.accuracy_score(y_train, y_predict)
print "f_measure:", metrics.f1_score(y_train, y_predict, average ='macro')
print "precision:", metrics.precision_score(y_train, y_predict, average = 'macro')
print "recall:", metrics.recall_score(y_train, y_predict, average = 'macro')

