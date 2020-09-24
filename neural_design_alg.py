# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:31:27 2020

Neural networks design
@author: aspod
"""

import numpy as np 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
#%%
"""
Sampling different train/test datasets
"""
from sklearn.model_selection import train_test_split

#loading data
df_gases_train_only = pd.read_csv("data_gases_train_only.csv")
df_gases_with_C = pd.read_csv("data_gases_with_C.csv")
df_gas_and_temp_train_only = pd.read_csv("data_test_gas_and_temp_train_only.csv")
df_gas_and_temp_with_C = pd.read_csv("data_test_gas_and_temp_with_C.csv")
df_stitch_1 = pd.read_csv("data_stitch_1.csv")
df_stitch_2 = pd.read_csv("data_stitch_2.csv")
#%%
#only gases data 

additional_nonezero_elms = df_gases_with_C.loc[df_gases_with_C['C_target'] != 0].sample(frac = 0.65, replace = True, random_state=42) 
df_gases_train, df_gasses_test = train_test_split(df_gases_with_C, random_state=42, test_size=0.4, shuffle=True)
additional_nonezero_elms.drop(columns='C', inplace=True)
df_gases_train.drop(columns='C', inplace=True)
df_gasses_only = pd.concat([df_gases_train,additional_nonezero_elms,df_gases_train_only])
#%%
#gases + small sample of stitch_1

df_stitch_1_train_small = df_stitch_1.sample(frac = 0.2, random_state=42)
df_gasses_and_stitch_small = pd.concat([df_gasses_only,df_stitch_1_train_small])

#gases + big sample of stitch_1
df_stitch_1_train_big = df_stitch_1.sample(frac = 0.5, random_state=42)
df_gasses_and_stitch_big = pd.concat([df_gasses_only,df_stitch_1_train_big])

#%%
#gases + sample of temp test (zeros)
df_gas_and_temp_with_C_train, df_gas_and_temp_with_C_test = train_test_split(df_gas_and_temp_with_C, random_state=42, test_size=0.25, shuffle=True)
df_gas_and_temp_with_C_train = df_gas_and_temp_with_C_train.sample(frac = 0.55, random_state=42)
df_gas_and_temp_with_C_train.drop(columns='C', inplace=True)
df_gases_and_temp = pd.concat([df_gasses_only,df_gas_and_temp_with_C_train])

#gases + gas(not zeros) + temp
df_gas_and_temp_train_only = df_gas_and_temp_train_only.loc[df_gas_and_temp_train_only['C_target'] != 0].sample(frac = 0.65, random_state=42)
df_gases_and_temp_gas = pd.concat([df_gases_and_temp,df_gas_and_temp_train_only])
#%%
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
#%%
"""
Main neural network pipeline
accept dataset for training
produce features and isolate target
split data into train and validate features and targets
start process of building and running networks
"""
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def nn_pipeline(df):
    features = df[['Um','Ur','Ud']]
    target = df['C_target']
    features = set_nonelinear_features(features)
    features_train, features_val, target_train, target_val = train_test_split(features, target,
                                                                          test_size=0.15, random_state=42, shuffle=True)
    scaler = preprocessing.StandardScaler().fit(features_train)
    features_train = scaler.transform(features_train) 
    features_val = scaler.transform(features_val)
    
    target_train = target_train.to_numpy()
    target_val = target_val.to_numpy()
    
    print(features_train.shape)
    
    return optimal_network(features_train, features_val, target_train, target_val)
#%%
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.core import Dense
from keras import regularizers
from keras.layers import Dropout
from keras.layers import BatchNormalization

def optimal_network(features_train, features_val, target_train, target_val):
    input_size = features_train.shape[1]
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    best_one_l_model = dict({'model':None, 'history':None, 'name':None})
    best_two_l_model = dict({'model':None, 'history':None, 'name':None})
    #variating learning rate of adam optimizer
    learning_rates = [0.05,0.01,0.001]
    for learning_rate in learning_rates:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #variating L2 regularizer coef
        reg_coefs = [0.01,0.001,0.0001]
        for reg_coef in reg_coefs:
            #building one layer models
            for neurs_numb in range(input_size-4, input_size*2 + 4, 4):
                one_layer_model =  Sequential([
                        Dense(neurs_numb, activation='relu', input_shape=(input_size,), kernel_regularizer=regularizers.l2(reg_coef)),
                        Dense(1)
                        ])
                one_layer_model.compile(loss='mse',
                                        optimizer=optimizer,
                                        metrics=['mse'])
                one_layer_model_name = f"one layer model nn = {neurs_numb}"
                history = one_layer_model.fit(features_train, target_train, epochs=2000, batch_size = 200, shuffle=True,
                    validation_data=(features_val, target_val), verbose=0, callbacks=[early_stop])
                if(best_one_l_model['model'] == None):
                    best_one_l_model['model'] = one_layer_model
                    best_one_l_model['name'] = one_layer_model_name
                    best_one_l_model['history'] = history
                else:
                    if best_one_l_model['history'].history['val_mse'][-1] > history.history['val_mse'][-1]:
                        best_one_l_model['model'] = one_layer_model
                        best_one_l_model['name'] = one_layer_model_name
                        best_one_l_model['history'] = history
                
            #building two layer models
            for neurs_1_L_numb in range(input_size, input_size*2 + 6, 6):
                for neurs_2_L_numb in range(neurs_1_L_numb + 12, neurs_1_L_numb - 12, -6):
                    two_layers_model =  Sequential([
                                        Dense(neurs_1_L_numb, activation='relu', input_shape=(input_size,), kernel_regularizer=regularizers.l2(reg_coef)),
                                        Dense(neurs_2_L_numb, activation='relu', kernel_regularizer=regularizers.l2(reg_coef)),
                                        Dense(1)
                                        ])
                    two_layers_model.compile(loss='mse',
                                        optimizer=optimizer,
                                        metrics=['mse'])
                    two_layers_model_name = f"two layer model nn 1 = {neurs_1_L_numb}, nn 2 = {neurs_2_L_numb}"
                    history = two_layers_model.fit(features_train, target_train, epochs=2000, batch_size = 200, shuffle=True,
                    validation_data=(features_val, target_val), verbose=0, callbacks=[early_stop])
                    if(best_two_l_model['model'] == None):
                        best_two_l_model['model'] =two_layers_model
                        best_two_l_model['name'] = two_layers_model_name
                        best_two_l_model['history'] = history
                    else:
                        if best_two_l_model['history'].history['val_mse'][-1] > history.history['val_mse'][-1]:
                            best_two_l_model['model'] = two_layers_model
                            best_two_l_model['name'] = two_layers_model_name
                            best_two_l_model['history'] = history    
            
            print(f"best 1 L model = {best_one_l_model['name']}\nbest 2 L model = {best_two_l_model['name']}")
        
        print(f"first {learning_rate} rate passes")    
    return best_one_l_model, best_two_l_model            
            
        
    
#%%
"""
Creating models on gases data only
"""
best_one_l_model_gasses_only, best_two_l_model_gasses_only = nn_pipeline(df_gasses_only)
#%%
best_one_l_model_gasses_only.save("best_one_l_model_gasses_only")
best_two_l_model_gasses_only.save("best_two_l_model_gasses_only")
#%%

best_one_l_model_gasses_and_stitch_small, best_two_l_model_gasses_and_stitch_small = nn_pipeline(df_gasses_and_stitch_small)
best_one_l_model_gasses_and_stitch_big, best_two_l_model_gasses_and_stitch_big = nn_pipeline(df_gasses_and_stitch_big)
best_one_l_model_gasses_and_temp, best_two_l_model_gasses_and_temp = nn_pipeline(df_gases_and_temp)
best_one_l_model_gasses_and_temp_gas, best_two_l_model_gasses_and_temp_gas = nn_pipeline(df_gases_and_temp_gas)


#%%
from matplotlib import  pyplot as plt

color_map =['red','blue','purple', 'green', 'black', 'yellow','brown', 'grey', 'orange','plum' , 'khaki', 'indigo']
def plot_history(models, xlim):    
    plt.figure(figsize=(40,20))
    i = 0
    for model in models:
        model_name = model['name']
        hist = model.history
        epoch = hist.epoch
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(epoch, hist['mse'], ms=1.5, color=color_map[i],
           label=f'{model_name} train')
        plt.plot(epoch, hist['val_mse'], color=color_map[i],
           label = f'{model_name} val', ms=1.5, linestyle = '--')
        plt.ylim([0,3000000])
        plt.xlim([0,xlim])
        i+=1
    plt.legend(fontsize=24, markerscale=10)
    plt.show()
#%%