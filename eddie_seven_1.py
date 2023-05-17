# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:52:34 2023

@author: abatiste
"""

#OPEN ANACONDA CMD AND ENTER 

#cd Desktop\Batiste1\ml_dl_deploy

#ENTER streamlit run eddie_seven_1.py

from sklearn.svm import SVC
import lap_testing4
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
import sklearn.metrics as metrics

import streamlit as st
import pandas as pd
import numpy as np
import base64

import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
import joblib

from prediction import predict

#use basic py to open and run your regression model and fill out the all_done df
exec(open('lap_testing4.py').read())

#filter for street legal weapons
all_done2 = all_done[all_done['STREET_LEGAL']=='Y']

model = SVC()

X = pd.DataFrame(all_done2, columns=['HP','WEIGHT','SKID', 'TRAP'])
y= pd.DataFrame(all_done2, columns=['GARAGE'])

X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size = .5, random_state =3)

model.fit(X_train, y_train)

class_predict = model.predict(X_test)

metrics.accuracy_score(y_test, class_predict) #on 15may23, i get .756 accuracy using hp/trap/skid/weight for garage

###############################################
###############################################

joblib.dump(model, 'svc_model.pkl')

joblib.dump(list(X.columns), 'svc_names.pkl')

##loading model

new_columns = joblib.load('svc_names.pkl')

new_columns

loaded_model = joblib.load('svc_model.pkl')

###############################################
###############################################
###############################################
###############################################




















