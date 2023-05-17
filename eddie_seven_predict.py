# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:14:20 2023

@author: abatiste
"""

import joblib

def predict(data):
    clf = joblib.load('svc_model.pkl')
    return clf.predict(data)
