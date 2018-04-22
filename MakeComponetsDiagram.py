# -*- coding: utf-8 -*-
"""
Spyder Editor
 
This is a temporary script file.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path
import csv
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_predict , GridSearchCV
import pandas as pd
from wordcloud import STOPWORDS 
from gensim.parsing.porter import PorterStemmer
import nltk, re, string
from nltk.stem import WordNetLemmatizer
from urllib import urlopen
from nltk.corpus import wordnet
from nltk import word_tokenize
from collections import Counter 

#from nltk.corpus import wordnet # To get words in dictionary with their parts of speech
#from nltk.stem import WordNetLemmatizer
'''
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
'''
 
def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]

train_data = pd.read_csv('train_set.csv' , sep = "\t", encoding = 'utf-8')
 
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

texts =[]
currentText = []
wnl = WordNetLemmatizer()

#Stemming
for whichtexts in train_data['Content']:
    #print "------------------------------------"
    #print whichtexts
    
    whichtexts = word_tokenize(whichtexts)
    for word in whichtexts:
        skata = wnl.lemmatize( word, get_pos(word) )
        currentText.append(skata)
    
    #print currentText
    #whichtexts = nltk.word_detokenize(currentText)
    whichtexts = " ".join(currentText)
    whichtexts = whichtexts.encode("ascii", errors ="ignore")
    whichtexts = p.stem_sentence(whichtexts)
    #print whichtexts
    texts.append(whichtexts)
    currentText = []
    
#Vectorazation

'''
for i in texts:
    print i
    print "-------------------------------------"
'''

count_vectorizer = CountVectorizer(stop_words=stopwords)
X_train = count_vectorizer.fit_transform(texts)


temp =X_train

with open('LSI_diagram.csv', 'w') as csvfile:
    
    
    data = {'LSI_Components': [100, 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115], 'Accuracy': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] }
    df = pd.DataFrame(data, columns=['LSI_Components', 'Accuracy'])
    #fieldnames = ['LSI_Components', 'Accuracy']
    #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #writer.writeheader()

    for i in range(100,115):
        #LSI
        svd = TruncatedSVD(n_components=i)
        X_train = svd.fit_transform(X_train)
            
            
            
        svc = svm.SVC(kernel='linear', C=0.01)
        y_predict = cross_val_predict(svc,X_train,y_train,cv=10)
            
            
        acc = metrics.accuracy_score(y_train, y_predict)
        print "accuracy:" , acc
        print "f_measure:", metrics.f1_score(y_train, y_predict, average ='macro')
        print "precision:", metrics.precision_score(y_train, y_predict, average = 'macro')
        print "recall:", metrics.recall_score(y_train, y_predict, average = 'macro')
            
        #writer.writerow({'LSI_Components': i, 'Accuracy': metrics.accuracy_score(y_train, y_predict)})
        df.loc[[i-100],['Accuracy']]=acc
        X_train = temp
    
    df.to_csv('SVM_LSI_diagram.csv')
        
    
    
    
    
    
    
