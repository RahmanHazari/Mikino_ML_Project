# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:24:01 2023

@author: Admin
"""
'''
###################     Data Dictionary  ##########################
Dataset contains makino machine details for unplaned brakdown
16 features are recorded'''

'''
##################       Description     #######################


* Date :The date at which the observation was made.

* Machine_ID:Unique Id of every machine.

*Assembly_Line_No:where a particular observation or event occurred.

*Hydraulic_Pressure(bar): Uniform hydraulic pressure in fluids.

*Coolant_Pressure(bar):Pressure of the coolant circulating through the cooling system.

*Air_System_Pressure(bar): Air brakes use compressed air to stop vehicles.

*Coolant_Temperature(°C): Temp of the liquid that circulates through the engine's
 cooling system to regulate its temperature.

*Hydraulic_Oil_Temperature(°C):Temp of the oil used in the hydraulic system.

*Spindle_Bearing_Temperature(°C):Temp of the bearings that support the 
spindle in the engine's rotating components.

*Spindle_Vibration(µm):amount of vibration produced by the spindle 
and its associated rotating components.

*Tool_Vibration(µm):amount of vibration produced by the cutting 
tools used in manufacturing engine components

*Spindle_Speed(RPM):Rotational speed of the spindle and its associated components.

*Voltage(volts):Electrical potential difference between two points
 in the engine's electrical system.

*Torque(Nm):Twisting force produced by the engine's crankshaft, 
which drives the wheels through the transmission.

*Cutting(kN):Monitoring cutting force to optimize the machining 
process for better quality and efficiency.

*Downtime: It is the target variable which tells about whether machine 
is going to fail or not (Downtime or No Downtime).
'''


# IMPORTING REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
import pickle, joblib
import sweetviz as sv
from feature_engine.outliers import Winsorizer
from feature_engine import transformation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
#from sqlalchemy import create_engine, text
from urllib.parse import quote 
from sklearn.metrics import accuracy_score, f1_score

# IMPORTING DATASET
data = pd.read_csv('D:\Data sets\data_makino.csv')


# INSIGHTS FROM DATA
data.shape
data.columns
data.info() # Null values are present in many columns
decs = data.describe() # Scaling to be done
data.Downtime.value_counts() # 2 class problem
data.drop(['Date', 'Machine_ID', 'Assembly_Line_No'], axis = 1, inplace = True)

# RENAMING COLUMNS
dict = {'Hydraulic_Pressure(bar)' : 'Hydraulic_Pressure_bar',
        'Coolant_Pressure(bar)' : 'Coolant_Pressure_bar',
        'Air_System_Pressure(bar)' : 'Air_System_Pressure_bar',
        'Coolant_Temperature(°C)' : 'Coolant_Tempearture_DegCelcius',
        'Hydraulic_Oil_Temperature(°C)' : 'Hydraulic_Oil_Temperature_DegCelcius',
        'Spindle_Bearing_Temperature(°C)' : 'Spindle_Bearing_Temperature_DegCelcius',
        'Spindle_Vibration(µm)' : 'Spindle_Vibration_micromtr', 
        'Tool_Vibration(µm)' : 'Tool_Vibration_micromtr', 
        'Spindle_Speed(RPM)' : 'Spindle_Speed_RPM',
        'Voltage(volts)' : 'Voltage_Volts', 
        'Torque(Nm)' : 'Torque_Nanomtr', 
        'Cutting(kN)' : 'Cutting_Kilonewton'}
 
data.rename(columns = dict, inplace = True)

# EXPLORATORY DATA ANALYSIS
# FIRST MOMENT BUSINESS DECISION
data.mean()
data.median()
data.mode()

# SECOND MOMENT BUSINESS DECISION
data.var()
data.std()
data.min()
data.max()

# THIRD MOMENT BUSINESS DECISION
data.skew()

# FORTH MOMENT BUSINESS DECISION
data.kurt()

# DATA VISULIZATIONS
# BOX PLOT
data.plot(kind = "box", sharey = False, figsize = (16,8), subplots = True, colormap = 'rainbow')
plt.subplots_adjust(wspace = 1.5) # Outliers are present in all columns except in cutting column

# HISTOGRAM
data.hist(grid = False, figsize = (10,6),color = '#86bf91')
plt.show()

# AUTO EDA
eda = sv.analyze(data)
eda.show_html()

# SPLITTING INPUTS AND OUTPUT
X = data.drop('Downtime', axis=1)
y = data['Downtime']

# DATA PREPROCESSING
# IDENTIFING DUPLICATES
duplicate = data.duplicated() # Returns Boolean Series denoting duplicate rows.
duplicate
sum(duplicate) # No duplicates found

# ZERO OR NEAR ZERO VARIANCE
data.var()
data.var() == 0

# PIPELINE FOR TREATING MISSING VALUES
imp_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'median'))]) 
processed1 = imp_pipeline.fit(X)
processed1
joblib.dump(processed1, 'processed1')
X1 = pd.DataFrame(processed1.transform(X), columns = X.columns)
X1.isna().sum() # No missing values

# OUTLIERS ANALYSIS
out_pipeline = Pipeline([('outlier', Winsorizer(capping_method = "iqr", tail = "both", fold = 1.5))]) 
processed2 = out_pipeline.fit(X1)
processed2
joblib.dump(processed2, 'processed2')
X2 = pd.DataFrame(processed2.transform(X1), columns = X1.columns).convert_dtypes()
X2.plot(kind = "box", sharey = False, figsize = (16,8), subplots = True, colormap = 'rainbow')
plt.subplots_adjust(wspace = 1.5) # No outliers

# TRANSFORMATIONS
# Checking whether data is normally distributed
stats.probplot(X2.Hydraulic_Pressure_bar.astype(int), dist = "norm", plot = pylab)
stats.probplot(X2.Coolant_Pressure_bar.astype(int), dist = "norm", plot = pylab) # NON NORMALLY DISTRIBUTED
stats.probplot(X2.Air_System_Pressure_bar.astype(int), dist = "norm", plot = pylab) # NON NORMALLY DISTRIBUTED
stats.probplot(X2.Coolant_Tempearture_DegCelcius.astype(int), dist = "norm", plot = pylab) # NON NORMALLY DISTRIBUTED
stats.probplot(X2.Hydraulic_Oil_Temperature_DegCelcius.astype(int), dist = "norm", plot = pylab)
stats.probplot(X2.Spindle_Bearing_Temperature_DegCelcius.astype(int), dist = "norm", plot = pylab) 
stats.probplot(X2.Spindle_Vibration_micromtr.astype(int), dist = "norm", plot = pylab) # NON NORMALLY DISTRIBUTED
stats.probplot(X2.Tool_Vibration_micromtr.astype(int), dist = "norm", plot = pylab)
stats.probplot(X2.Spindle_Speed_RPM.astype(int), dist = "norm", plot = pylab) 
stats.probplot(X2.Voltage_Volts.astype(int), dist = "norm", plot = pylab)
stats.probplot(X2.Torque_Nanomtr.astype(int), dist = "norm", plot = pylab)
stats.probplot(X2.Cutting_Kilonewton.astype(int), dist = "norm", plot = pylab) # NON NORMALLY DISTRIBUTED
X3 = X2.astype(int)

# TRANSFORMING THE NON NORMAL DATA INTO NORMAL DATA
tf = transformation.YeoJohnsonTransformer()
X4 = tf.fit_transform(X3)

# CHECKING FOR NORMAL DISTRIBUTION AFTER TRANSFORMATION
stats.probplot(X4.Hydraulic_Pressure_bar, dist = "norm", plot = pylab)
stats.probplot(X4.Coolant_Pressure_bar, dist = "norm", plot = pylab)
stats.probplot(X4.Air_System_Pressure_bar, dist = "norm", plot = pylab)
stats.probplot(X4.Coolant_Tempearture_DegCelcius, dist = "norm", plot = pylab)
stats.probplot(X4.Hydraulic_Oil_Temperature_DegCelcius, dist = "norm", plot = pylab)
stats.probplot(X4.Spindle_Bearing_Temperature_DegCelcius, dist = "norm", plot = pylab) 
stats.probplot(X4.Spindle_Vibration_micromtr, dist = "norm", plot = pylab)
stats.probplot(X4.Tool_Vibration_micromtr, dist = "norm", plot = pylab)
stats.probplot(X4.Spindle_Speed_RPM, dist = "norm", plot = pylab) 
stats.probplot(X4.Voltage_Volts, dist = "norm", plot = pylab)
stats.probplot(X4.Torque_Nanomtr, dist = "norm", plot = pylab)
stats.probplot(X4.Cutting_Kilonewton, dist = "norm", plot = pylab)

# PIPELINE FOR SCALING
sca_pipeline = Pipeline([('scale', MinMaxScaler())]) 
processed3 = sca_pipeline.fit(X4)
processed3
joblib.dump(processed3, 'processed3')
X5 = pd.DataFrame(processed3.transform(X4), columns = X4.columns)
des = X5.describe()

# SPLITTING THE DATA INTO TRAIN & TEST
X_train, X_test, y_train, y_test = train_test_split(X5, y, test_size = 0.2, random_state = 0)
X_train.shape
X_test.shape



## Automate the model selection process 

import lazypredict
from lazypredict.Supervised import LazyClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.30, stratify=y,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

models


########## MODAL BUILDING ##########

##### RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# PREDICTING ON TEST DATA
pred_test = rfc.predict(X_test)  
pred_test

# CROSS TABLE FOR TEST DATA
pd.crosstab(y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TEST DATA
print(accuracy_score(y_test, pred_test))

# PREDICTING ON TRAIN DATA
pred_train = rfc.predict(X_train)  
pred_train

# CROSS TABLE FOR TRAIN DATA
pd.crosstab(y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TRAIN DATA
print(accuracy_score(y_train, pred_train))

# HYPERPARAMETER TUNING
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [2, 4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
bootstrap = [True, False]

# create a dictionary of all hyperparameters to be experimented
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# GridsearchCV with cross-validation to perform experiments with parameters set
rfc_gscv = GridSearchCV(rfc, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)

# Train
rfc_gscv.fit(X5, y)

# The best set of parameter values
rfc_gscv.best_params_

# Model with best parameter values
RFC_best = rfc_gscv.best_estimator_
RFC_best

# PREDICTING ON TEST DATA
pred_test = RFC_best.predict(X_test)  
pred_test

# CROSS TABLE FOR TEST DATA
pd.crosstab(y_test, pred_test, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TEST DATA
print(accuracy_score(y_test, pred_test))

# F1 SCORE FOR TEST
print(f1_score(y_test, pred_test, pos_label = 'Machine_Failure'))

# PREDICTING ON TRAIN DATA
pred_train = RFC_best.predict(X_train)  
pred_train

# CROSS TABLE FOR TRAIN DATA
pd.crosstab(y_train, pred_train, rownames = ['Actual'], colnames = ['Predictions']) 

# ACCURACY SCORE FOR TRAIN DATA
print(accuracy_score(y_train, pred_train))

# F1 SCORE FOR TRAIN
print(f1_score(y_train, pred_train, pos_label = 'Machine_Failure'))

# SAVING THE BEST MODEL
pickle.dump(RFC_best, open('RFC_best.pkl', 'wb'))

