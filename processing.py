# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:14:21 2020

@author: yatee
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re



def clean_data(data):
    clean_corpus = []
    stemmer = PorterStemmer()
    for sentance in data:
        sentance = re.sub('[^a-zA-Z]', ' ', sentance)
        li = []
        for word in sentance.split(' '):
            if word not in stopwords.words('english'):
                li.append(stemmer.stem(word))
        li = list(filter(lambda x: x != '', li))
        sentance = ' '.join(li)
        clean_corpus.append(sentance)
    return np.array(clean_corpus)

def generate():
    raw_data = pd.read_csv('./data/SMSSpamCollection', delimiter='\t', names=['spam', 'sms'])
    
    vectorizer = TfidfVectorizer(lowercase='False', encoding='utf-8')
    cleaned_data = clean_data(raw_data['sms'])
    vector_model = vectorizer.fit(cleaned_data)
    
    with open('./vectoriser.pickle', 'wb') as f:
        pickle.dump(vector_model, f, protocol=2)
        print('vectorizer generated')
    
    x_data = vectorizer.transform(cleaned_data)
    
    y_dummies = pd.get_dummies(raw_data['spam'])
    y_data = y_dummies['spam']
    
    model = RandomForestClassifier().fit(x_data, y_data)
    
    with open('./model.pickle', 'wb') as f:
        pickle.dump(model, f, protocol=2)
        print('model generated')
if __name__=='__main__':
    generate()