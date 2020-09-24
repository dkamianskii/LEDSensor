# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:16:41 2020

@author: Dmitrii
"""
#%%
"""
Loading libs
"""

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%%
"""
Loading dataframes
"""

df_gases_1 = pd.read_pickle("./data_gases_1.pkl")
df_gases_2 = pd.read_csv("./data_gases_2.pkl")
df_stitch_1 = pd.read_pickle("./data_stitch_1.pkl")
df_stitch_2 = pd.read_pickle("./data_stitch_2.pkl")
#%%
"""
Filtering outliers
"""

#%%