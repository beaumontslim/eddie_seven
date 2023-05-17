# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:21:37 2023

@author: abatiste
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from eddie_seven_predict import predict

#OPEN ANACONDA CMD AND ENTER 

#cd Desktop\Batiste1\ml_dl_deploy

#ENTER streamlit run eddie_seven_2.py

#use basic py to open and run your regression model and fill out the all_done df
#exec(open('lap_testing4.py').read())

#filter for street legal weapons
#all_done2 = all_done[all_done['STREET_LEGAL']=='Y']

#exec(open('lap_testing4.py').read())

#df = all_done2
#st.dataframe(df)

st.title('SVC for Hoons')
st.markdown('Vector Machine model using Eddie Seven data')

st.header('Weapon Features')
col1, col2 = st.columns(2)
with col1:
    st.text('Static Stats')
    HP = st.slider('Power', 100, 700, 100)
    WEIGHT = st.slider('Weight (lbs)', 1500, 8000, 1500)
with col2:
    st.text('Dyanmic Stats')
    SKID = st.slider('Lateral Gs', 0.5, 1.5, 0.5)
    TRAP = st.slider('1/4 TRAP SPEED', 65, 180, 65)   

if st.button('Yes or No!'):
    result = predict(np.array([[HP, WEIGHT, SKID, TRAP]]))
    st.text(result[0])

###############################################
###############################################






