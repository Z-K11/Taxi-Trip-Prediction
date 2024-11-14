from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import gc,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
# Features
# tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,RatecodeID,store_and_fwd_flag,PULocationID,DOLocationID,
# payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount,congestion_surcharge
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
plt.hist(cleanData.tip_amount.values,16,histtype="bar",facecolor='g')
#plt.savefig("Visual_Representation.png")
print("Minimum amount is ", np.min(cleanData.tip_amount.values))
print("Maximum amount is ", np.max(cleanData.tip_amount.values))
print("90% of the trips have a tip amount less or equal than :",np.percentile(cleanData.tip_amount.values,90))
cleanData['tpep_pickup_datetime'] = pd.to_datetime(cleanData['tpep_pickup_datetime'])
# converts the data into pandas date_time object
cleanData["tpep_pickup_datetime"] = pd.to_datetime(cleanData['tpep_pickup_datetime'])
