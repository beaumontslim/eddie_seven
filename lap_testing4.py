# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:14:29 2021

@author: abatiste
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from pandasql import sqldf


####################################################################################################################################

#lap_testing = pd.read_excel('C:/Users/abatiste/ml_dl_deploy/lap_testing.xlsx')

lap_testing = pd.read_excel('C:/Users/abatiste/Desktop/lap_testing1.xlsx')

#lap_testing = lap_testing[(lap_testing['STREET_LEGAL']=='Y')]

#convert data types

lap_testing['CYLINDERS'] = lap_testing['CYLINDERS'].apply(str)
lap_testing['YEAR'] = lap_testing['YEAR'].apply(str)

#impute nans for brake distances
                       
impute_sixty = lap_testing['BSIXTY1'].mean()/lap_testing['BSEVENTY1'].mean()
impute_seventy = lap_testing['BSEVENTY1'].mean()/lap_testing['BSIXTY1'].mean()

lap_testing['BSIXTY1'] = lap_testing['BSIXTY1'].fillna(lap_testing['BSEVENTY1']*impute_sixty)
lap_testing['BSEVENTY1'] = lap_testing['BSEVENTY1'].fillna(lap_testing['BSIXTY1']*impute_seventy)

#lap_testing['SKID'] = lap_testing['SKID'].fillna(lap_testing.groupby('TYPE')['SKID'].transform('mean'))

lap_testing.info()

#lets work on fig eight data prep

lap_8 = lap_testing.dropna(subset=['FIGEIGHT'])

#prep to build model for figeight

car_object = lap_8.select_dtypes(include='object')
car_numeric = lap_8.select_dtypes(exclude='object')

X = car_numeric.drop(['RING', 'PRICE','LAGUNA', 'FIGEIGHT'], axis = 1)
y = car_numeric['FIGEIGHT']

####try different models...

#1
model = Ridge(alpha = .05)
model.fit(X, y)

#2
#model = LassoCV(eps = 0.001, n_alphas = 100, cv=  10)
#model.fit(X, y)

#3
#model = RidgeCV(alphas =(.1, 1.0, 10.0), scoring = 'neg_mean_absolute_error')
#model.fit(X, y)

#4
#model = RandomForestRegressor(n_estimators=30, random_state=101)
#model.fit(X, y)

scores_cross_val_score = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = 10)
scores_cross_val_score

abs(scores_cross_val_score.mean())

Xnew = [[112,	505,	3956,	3.5,	11.6,	123.37,	1.04,	142.2,	100]]

model.predict(Xnew)

nan_8 = lap_testing[lap_testing['FIGEIGHT'].isnull()]

nan_8['BSEVENTY1'] = nan_8['BSEVENTY1'].fillna(nan_8.groupby('TYPE')['BSEVENTY1'].transform('mean'))
nan_8['BSIXTY1'] = nan_8['BSIXTY1'].fillna(nan_8.groupby('TYPE')['BSIXTY1'].transform('mean'))
nan_8['QUARTER'] = nan_8['QUARTER'].fillna(nan_8.groupby('TYPE')['QUARTER'].transform('mean'))
nan_8['TRAP'] = nan_8['TRAP'].fillna(nan_8.groupby('TYPE')['TRAP'].transform('mean'))
nan_8['SKID'] = nan_8['SKID'].fillna(nan_8.groupby('TYPE')['SKID'].transform('mean'))


nan_8['FIGEIGHT'] = model.predict(nan_8[['WHEELBASE','HP','WEIGHT','ACCEL','QUARTER', 'TRAP', 'SKID','BSEVENTY1','BSIXTY1']])

frames = [lap_8, nan_8]

lap8_done  = pd.concat(frames)

#######################################
###############laguna

laguna_time = lap8_done.dropna(subset=['LAGUNA'])

car_object = laguna_time.select_dtypes(include='object')
car_numeric = laguna_time.select_dtypes(exclude='object')

#x and y, train  model

X = car_numeric.drop(['RING','PRICE','LAGUNA'], axis = 1)
y = car_numeric['LAGUNA']

#1
model = Ridge(alpha = .01)
model.fit(X, y)

#2
#model = LassoCV(eps = 0.001, n_alphas = 100, cv=  10)
#model.fit(X, y)

#3
#model = RidgeCV(alphas =(.1, 1.0, 10.0), scoring = 'neg_mean_absolute_error')
#model.fit(X, y)

#4
#model = RandomForestRegressor(n_estimators=30, random_state=101)
#model.fit(X, y)


scores_cross_val_score = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = 10)
scores_cross_val_score

abs(scores_cross_val_score.mean())

#test. will it make sense? (hint, doing very well on a few cars i tested)

Xnew = [[110.82,	500,	3900,	3.6,	11.5,	126,	1.06,	136,	100,	23.2]]

model.predict(Xnew)

laguna_null = lap8_done[lap8_done['LAGUNA'].isnull()]

laguna_null.info()

laguna_null['BSEVENTY1'] = laguna_null.groupby('TYPE')['BSEVENTY1'].transform(lambda value: value.fillna(value.mean()))
laguna_null['BSIXTY1'] = laguna_null.groupby('TYPE')['BSIXTY1'].transform(lambda value: value.fillna(value.mean()))
laguna_null['QUARTER'] = laguna_null.groupby('TYPE')['QUARTER'].transform(lambda value: value.fillna(value.mean()))
laguna_null['TRAP'] = laguna_null.groupby('TYPE')['TRAP'].transform(lambda value: value.fillna(value.mean()))
laguna_null['SKID'] = laguna_null.groupby('TYPE')['SKID'].transform(lambda value: value.fillna(value.mean()))


laguna_null['LAGUNA'] = model.predict(laguna_null[['WHEELBASE','HP','WEIGHT','ACCEL','QUARTER', 'TRAP', 'SKID','BSEVENTY1','BSIXTY1', 'FIGEIGHT']])

#finish laguna

frames = [laguna_time, laguna_null]

laguna_done  = pd.concat(frames)

###### try to finish the fight...########################################

ring_time = laguna_done.dropna(subset=['RING'])

ring_time.info()

car_object = ring_time.select_dtypes(include='object')
car_numeric = ring_time.select_dtypes(exclude='object')

#x and y, train  model

X = car_numeric.drop(['PRICE','RING'], axis = 1)
y = car_numeric['RING']

#1
model = Ridge(alpha = 7777)
model.fit(X, y)

#2
#model = LassoCV(eps = 0.01, n_alphas = 100, cv=  72)
#model.fit(X, y)

#3
#model = RidgeCV(alphas =(.1, 1.0, 10.0), scoring = 'neg_mean_absolute_error')
#model.fit(X, y)

#4
model = RandomForestRegressor(n_estimators=30, random_state=101)
model.fit(X, y)

scores_cross_val_score = cross_val_score(model, X, y, scoring = 'neg_mean_squared_error', cv = 10)
scores_cross_val_score

abs(scores_cross_val_score.mean())

scores_cross_val_score.mean()

#test. will it make sense? (hint, yes)

Xnew = [[106.3,505	,3890,	3.45,11.5,	126.4,	1.05,140,100,23.5,96]]

model.predict(Xnew)

#lets do ring now

ring_null = laguna_done[laguna_done['RING'].isnull()]

ring_null['RING'] = model.predict(ring_null[['WHEELBASE','HP','WEIGHT','ACCEL','QUARTER', 
                                            'TRAP', 'SKID','BSEVENTY1','BSIXTY1', 'FIGEIGHT', 'LAGUNA']])

#finish ring

frames = [ring_time, ring_null]

all_done  = pd.concat(frames)

########

#build dataframe for model output vs actual times? how many are within a few seconds? the ring is a long track 

ring_time['RING_MOD'] = model.predict(ring_time[['WHEELBASE','HP','WEIGHT','ACCEL','QUARTER', 
                                            'TRAP', 'SKID','BSEVENTY1','BSIXTY1', 'FIGEIGHT', 'LAGUNA']])

ring_time['DELTA'] = ring_time['RING']-ring_time['RING_MOD']

ring_time['DELTA'].abs().mean()

#test for how many are within 5 seconds?

(abs(ring_time['DELTA'])<=10).value_counts()[True]/len(ring_time)

#check to see how many are within 3 percent of the real time?

(abs(ring_time['DELTA'])<=(ring_time['RING'].mean()*.05)).value_counts()[True]/len(ring_time)

#4 column df so that we can easily view the deltas and the cars next to each other

ring_time_compare  = ring_time[['YEAR','MODEL','RING', 'RING_MOD', 'DELTA']]

#all_done.to_excel('all_done.xlsx', index = False)




















