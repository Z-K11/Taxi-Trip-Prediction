from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import gc,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler

data = pd.read_csv('yellow_tripdata_2019-06.csv')
# print(data.head())
print("There are " + str(len(data)) + " observations in the dataset.")
print("There are " + str(len(data.columns)) + " variables in the dataset.")
data = data[data['tip_amount']>0]
data = data[(data['tip_amount']<=data['fare_amount'])]
data = data[((data['tip_amount']>=2)&(data['tip_amount']<200))]
cleanData = data.drop(['total_amount'],axis= 1)
# The axis=1 parameter indicates that the drop should happen along columns (rather than rows).
del data
# frees the memory
gc.collect()
# is used to trigger manual garbage collection which involves clearing up unused objects and space hence freeing up memory
print("There are " + str(len(cleanData)) + " observations in the dataset.")
print("There are " + str(len(cleanData.columns)) + " variables in the dataset.")