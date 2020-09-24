# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:35:12 2020

Filtering outliers
@author: Dmitrii
"""

import numpy as np
import pandas as pd

#%%
def get_left_right_border(df, border_param):
    df_gases_grouped = df.groupby('C_target')
    quar_1 = df_gases_grouped[border_param].quantile(0.25)
    quar_3 = df_gases_grouped[border_param].quantile(0.75)
    inter_quar = {perc: (quar_3.get(perc) - quar_1.get(perc)) for perc in quar_1.keys()} 
    left_right_border = {perc: (quar_1.get(perc) - 1.5*inter_quar.get(perc),
                            quar_3.get(perc) + 1.5*inter_quar.get(perc)) for perc in quar_1.keys()}
    return left_right_border
#%%
def filter_outliers(df, left_right_border, border_param):
    df_filtered = df.copy()
    for target in left_right_border.keys():
        df_filtered.drop(df_filtered[(df_filtered['C_target'] == target) &
                                   ((df_filtered[border_param] < left_right_border.get(target)[0]) |
                                    (df_filtered[border_param] > left_right_border.get(target)[1]))].index, inplace=True) 
    return df_filtered
#%%
df_test_gas_and_temp_1 = pd.read_csv("data_test_gas_and_temp_1.csv")
df_test_gas_and_temp_2 = pd.read_csv("data_test_gas_and_temp_2.csv")

#%%
left_right_border_gas_and_temp_1 = get_left_right_border(df_test_gas_and_temp_1, 'Um')
left_right_border_gas_and_temp_2 = get_left_right_border(df_test_gas_and_temp_2, 'Um')

#%%
df_test_gas_and_temp_filt_1 = filter_outliers(df_test_gas_and_temp_1, left_right_border_gas_and_temp_1, 'Um')
df_test_gas_and_temp_filt_2 = filter_outliers(df_test_gas_and_temp_2, left_right_border_gas_and_temp_2, 'Um')
#%%
df_test_gas_and_temp_filt_1.to_csv("data_test_gas_and_temp_1_filtered.csv")
df_test_gas_and_temp_filt_2.to_csv("data_test_gas_and_temp_2_filtered.csv")
#%%
df_gases_1 = pd.read_csv("data_gases_1.csv")
df_gases_2 = pd.read_csv("data_gases_2.csv")

left_right_border_gases_1 = get_left_right_border(df_gases_1, 'Um')
left_right_border_gases_2 = get_left_right_border(df_gases_2, 'Um')

df_gases_1_filtered = filter_outliers(df_gases_1, left_right_border_gases_1, 'Um')
df_gases_2_filtered = filter_outliers(df_gases_2, left_right_border_gases_2, 'Um')

df_gases_1_filtered.to_csv("data_gases_1_filtered.csv")
df_gases_2_filtered.to_csv("data_gases_2_filtered.csv")
#%%
"""
Filtering Stitch data
"""

df_stitch_1 = pd.read_csv("data_stitch_1.csv")
df_stitch_2 = pd.read_csv("data_stitch_2.csv")

#%%
df_gases_1.drop(columns='C', inplace=True)
df_test_gas_and_temp_1_with_C = df_test_gas_and_temp_1.loc[~df_test_gas_and_temp_1['C'].isnull()]
df_test_gas_and_temp_2_with_C = df_test_gas_and_temp_2.loc[~df_test_gas_and_temp_2['C'].isnull()]
df_test_gas_and_temp_2_train_only = df_test_gas_and_temp_2.loc[df_test_gas_and_temp_2['C'].isnull()]
df_test_gas_and_temp_1_train_only = df_test_gas_and_temp_1.loc[df_test_gas_and_temp_1['C'].isnull()]
df_test_gas_and_temp_1_train_only.drop(columns='C', inplace=True)
df_test_gas_and_temp_2_train_only.drop(columns='C', inplace=True)
#%%
df_test_gas_and_temp_with_C = pd.concat([df_test_gas_and_temp_1_with_C,df_test_gas_and_temp_2_with_C])
df_test_gas_and_temp_train_only = pd.concat([df_test_gas_and_temp_1_train_only,df_test_gas_and_temp_2_train_only])
#%%
df_gases_1.to_csv("data_gases_train_only.csv")
df_test_gas_and_temp_with_C.to_csv("data_test_gas_and_temp_with_C.csv") 
df_test_gas_and_temp_train_only.to_csv("data_test_gas_and_temp_train_only.csv")