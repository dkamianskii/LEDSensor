# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:06:02 2020

@author: aspod
"""


import numpy as np 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_gases_with_C = pd.read_csv("data_gases_with_C.csv")

def set_nonelinear_features(features):
    features['Um^2'] = features['Um'].pow(2)
    features['Um^3'] = features['Um'].pow(3)
    features[f'ln(Um)'] = np.log(features['Um'])
    features['Ur^2'] = features['Ur'].pow(2)
    features['Ur^3'] = features['Ur'].pow(3)
    features['Ud^2'] = features['Ud'].pow(2)
    features['Ud^3'] = features['Ud'].pow(3)
    features['Um/Ur'] = features['Um']/features['Ur']
    features['Um/Ud'] = features['Um']/features['Ud']
    return features

def test_model(df_test, model, scaler):
    C_target_test = df_test['C_target']
    predicts = []
    feats = []
    for target in sorted(set(C_target_test)):
        temp_test = df_test[df_test['C_target'] == target]
        C_original = temp_test['C']
        C_target = temp_test['C_target']
        temp_features = temp_test[['Um','Ur','Ud']].copy()
        temp_features = set_nonelinear_features(temp_features)
        feats.append(temp_features)
        temp_features = scaler.transform(temp_features)
        temp_predict = model.predict(temp_features,verbose=0)
        predicts.append(temp_predict)

    return predicts,feats

#%%
from keras.models import load_model
import pickle

model = load_model('one layer model nn = 16 gases_and_temp',compile=False)
scaler = pickle.load(open('scaler_3.pkl','rb'))
predicts,feats = test_model(df_gases_with_C, model, scaler)