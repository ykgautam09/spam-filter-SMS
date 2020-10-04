# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:47:50 2020

@author: yatee
"""

import pickle
from processing import clean_data

with open('./model.pickle', 'rb') as f:
    model = pickle.load(f)
with open('./vectoriser.pickle', 'rb') as f:
    vectoriser = pickle.load(f)


def predict_class(sms):
    cleaned_sms = clean_data([sms])
    vector_sms = vectoriser.transform(cleaned_sms)
    output = model.predict(vector_sms)
    return output


if __name__ == '__main__':
    print(predict_class('WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'))
