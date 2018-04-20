# -*- coding: utf-8 -*-
"""
Spyder Editor
 
This is a temporary script file.
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from os import path
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
 
train_data = pd.read_csv('train_set.csv' , sep = "\t")
 
#train_data = train_data[0:25]
 
stopwords = set(ENGLISH_STOP_WORDS)
stopwords2 = set(STOPWORDS)
stopwords.union(stopwords2)
stopwords.add("said")
stopwords.add("say")
stopwords.add("says")
stopwords.add("will")

 
politics_texts = "" #id 3
film_texts = "" #id 1
football_texts = "" #id 2
businness_texts = "" #id 0
technology_texts = "" #id 4
 
le=preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y=le.transform(train_data["Category"])
i=0

for index, rows in train_data.iterrows():
    
    if y[i]==0:
        #append to business
        businness_texts+=" " + rows['Content']
    elif y[i]==1:
        #append to film
        film_texts+=" " + rows['Content']
    elif y[i]==2:
        #append to football
        football_texts+=" " + rows['Content']
    elif y[i]==3:
        #append to politics
        politics_texts+=" " + rows['Content']
    elif y[i]==4:
        #append to technology
        technology_texts+=" " + rows['Content']
    i+=1
 
d = path.dirname(__file__)
 
wordcloud_businness = WordCloud(stopwords=stopwords)
wordcloud_film = WordCloud(stopwords=stopwords)
wordcloud_football = WordCloud(stopwords=stopwords)
wordcloud_politics = WordCloud(stopwords=stopwords)
wordcloud_technology = WordCloud(stopwords=stopwords)
 
wordcloud_businness.generate(businness_texts)
wordcloud_film.generate(film_texts)
wordcloud_football.generate(football_texts)
wordcloud_politics.generate(politics_texts)
wordcloud_technology.generate(technology_texts)


wordcloud_businness.to_file(path.join(d, "wordcloud_businness.png"))
wordcloud_film.to_file(path.join(d, "wordcloud_film.png"))
wordcloud_football.to_file(path.join(d, "wordcloud_football.png"))
wordcloud_politics.to_file(path.join(d, "wordcloud_politics.png"))
wordcloud_technology.to_file(path.join(d, "wordcloud_technology.png"))
 

plt.suptitle('Businness WordCloud', fontsize = 20)
plt.imshow(wordcloud_businness, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.suptitle('Film WordCloud', fontsize = 20)
plt.imshow(wordcloud_film, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.suptitle('Football WordCloud', fontsize = 20)
plt.imshow(wordcloud_football, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.suptitle('Politics WordCloud', fontsize = 20)
plt.imshow(wordcloud_politics, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.suptitle('Technology WordCloud', fontsize = 20)
plt.imshow(wordcloud_technology, interpolation='bilinear')
plt.axis("off")
plt.show()