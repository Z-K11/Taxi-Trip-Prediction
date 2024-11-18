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
#print("There are " + str(len(cleanData)) + " observations in the dataset.")
#print("There are " + str(len(cleanData.columns)) + " variables in the dataset.")
#plt.hist(cleanData.tip_amount.values,16,histtype="bar",facecolor='g')
#plt.savefig("Visual_Representation.png")
print("Minimum amount is ", np.min(cleanData.tip_amount.values))
print("Maximum amount is ", np.max(cleanData.tip_amount.values))
print("90% of the trips have a tip amount less or equal than :",np.percentile(cleanData.tip_amount.values,90))
cleanData['tpep_pickup_datetime'] = pd.to_datetime(cleanData['tpep_pickup_datetime'])
# converts the data into pandas date_time object
cleanData["tpep_dropoff_datetime"] = pd.to_datetime(cleanData['tpep_dropoff_datetime'])
#print(cleanData.columns)
cleanData['pickup_hour']=(cleanData['tpep_pickup_datetime']).dt.hour
cleanData['dropoff_hour']=(cleanData['tpep_dropoff_datetime']).dt.hour
# the above code gives us pick up hour and drop off hour from date time column .dt.hour extract the hour(0-23_ from each timestamp in the column)
cleanData['pickup_day']=(cleanData['tpep_pickup_datetime']).dt.weekday
cleanData['dropoff_day']=(cleanData['tpep_dropoff_datetime']).dt.weekday
# the above code converts pick_up_datetime to by extracting the day as integer monday = 0 tuesday = 1
cleanData['trip_time']=(cleanData['tpep_dropoff_datetime']-cleanData['tpep_pickup_datetime']).dt.total_seconds
# .dt.total_seconds converts a time duration from a time delta object into a float representing the total number of seconds
# Idealy we would want to use the whole data set for this project 
# But if you run into memory issues you can use the first 200000 rows by uncommenting the below two lines
# reduce_data_set = 200000
# cleanData = cleanData.head(reduce_data_set)
cleanData = cleanData.drop(['tpep_pickup_datetime','tpep_dropoff_datetime'],axis =1 )
# this code will remove the pickup and dropoff time from the data set and restore the remaining data set into the assigned variable 
# some features are categorical, we need to encode them
# to encode them we use one-hot encoding from the Pandas package
get_dummy_col = ["VendorID","RatecodeID","store_and_fwd_flag","PULocationID", "DOLocationID","payment_type", "pickup_hour", "dropoff_hour", "pickup_day", "dropoff_day"]
proc_data = pd.get_dummies(cleanData,columns=get_dummy_col)
# release memory occupied by clean_data as we do not need it anymore
# we are dealing with a large dataset, thus we need to make sure we do not run out of memory
del cleanData
gc.collect()
# Extracting the label from the data frame called tip_amount
y = proc_data[['tip_amount']].values.astype('float32')
# droping tip_amount from the feature matrix
proc_data=proc_data.drop(['tip_amount'],axis=1)
X = proc_data.values
X = normalize(X,axis=1,norm=11,copy=False)
print('X.shape=', X.shape, 'y.shape=', y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
# print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)
# import the Decision Tree Regression Model from scikit-learn
#from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
#sklearn_dt = DecisionTreeRegressor(max_depth=8, random_state=35)

# train a Decision Tree Regressor using scikit-learn
#t0 = time.time()
#sklearn_dt.fit(X_train, y_train)
#sklearn_time = time.time()-t0
#print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))
# import the Decision Tree Regressor Model from Snap ML
#from snapml import DecisionTreeRegressor

# in contrast to sklearn's Decision Tree, Snap ML offers multi-threaded CPU/GPU training 
# to use the GPU, one needs to set the use_gpu parameter to True
# snapml_dt = DecisionTreeRegressor(max_depth=4, random_state=45, use_gpu=True)

# to set the number of CPU threads used at training time, one needs to set the n_jobs parameter
# for reproducible output across multiple function calls, set random_state to a given integer value
#snapml_dt = DecisionTreeRegressor(max_depth=8, random_state=45, n_jobs=4)

# train a Decision Tree Regressor model using Snap ML
#t0 = time.time()
#snapml_dt.fit(X_train, y_train)
#snapml_time = time.time()-t0
#print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))
# Snap ML vs Scikit-Learn training speedup
#training_speedup = sklearn_time/snapml_time
#print('[Decision Tree Regressor] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

# run inference using the sklearn model
#sklearn_pred = sklearn_dt.predict(X_test)

# evaluate mean squared error on the test dataset
#sklearn_mse = mean_squared_error(y_test, sklearn_pred)
#print('[Scikit-Learn] MSE score : {0:.3f}'.format(sklearn_mse))

# run inference using the Snap ML model
#snapml_pred = snapml_dt.predict(X_test)

# evaluate mean squared error on the test dataset
#snapml_mse = mean_squared_error(y_test, snapml_pred)
#print('[Snap ML] MSE score : {0:.3f}'.format(snapml_mse))