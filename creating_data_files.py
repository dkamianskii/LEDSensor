# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:49:05 2020

@author: Dmitrii
"""
#%%
import numpy as np
import pandas as pd
import matplotlib as plt
import scipy as sci
#%%
data_frames = pd.read_excel("rawData.xlsx", sheet_name=["stitch_1","stitch_2","gases_1","gases_2","temperature_1","temperature_2","all_night"], header=0)

data_frames['stitch_1']['C_target'] = 0.0
data_frames['stitch_2']['C_target'] = 0.0
data_frames['temperature_1'] = data_frames['temperature_1'][['C','Um','Ur','Ud']]
data_frames['temperature_2'] = data_frames['temperature_2'][['C','Um','Ur','Ud']]
data_frames['temperature_1']['C_target'] = 0.0
data_frames['temperature_2']['C_target'] = 0.0

def add_targets(df, perc_to_times):
    for ind, row in df.iterrows():
        for perc_to_time in perc_to_times:
            if (row['t'] < perc_to_time[0]) or (perc_to_time[0] == -1):
                df.loc[ind, "C_target"] = perc_to_time[1]*10000
                break
            

perc_to_times_1 = [(3042, 0.),( 4502,0.05),(6032,0.2),(7522,0.5),(8962, 1.),(10582, 2.), (12087, 3.), (-1, 5.)]
perc_to_times_2 = [(3730, 0.),( 5100,0.0498),(6300,0.254),(7475,2.05),(8695, 3.1),(9950, 5.), (-1, 0)]
add_targets(data_frames['gases_1'], perc_to_times_1)
add_targets(data_frames['gases_2'], perc_to_times_2)

data_frames['all_night'] = data_frames['all_night'][['C','Um','Ur','Ud']]
data_frames['all_night']['C_target'] = 1.8

data_frames['stitch_2'] = data_frames['stitch_2'][data_frames['stitch_2']['t'] < 8000]

data_frames['stitch_1'].drop(labels ='t', axis='columns', inplace=True)
data_frames['stitch_2'].drop(labels ='t', axis='columns', inplace=True)

data_frames['gases_1'].drop(labels ='t', axis='columns', inplace=True)
data_frames['gases_2'].drop(labels ='t', axis='columns', inplace=True)

for df_name in data_frames.keys():
    data_frames[df_name].to_csv(f"./data_{df_name}.csv")
    
