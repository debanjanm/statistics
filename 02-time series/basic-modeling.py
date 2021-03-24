# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:51:52 2021

@author: acer
"""

####################################################################################################################################
####################################################################################################################################
# importing required libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

####################################################################################################################################
####################################################################################################################################

print(os.getcwd())  # Prints the current working directory
path="C:/Users/acer/OneDrive/Documents/GitHub/statistics/02-time series"
os.chdir(path)

###############################################################################

DATA_DIR = "data" # indicate magical constansts (maybe rather put it on the top of the script)
ts_filename = "AirPassengers.csv"# fix gruesome var names
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(os.path.join(DATA_DIR, ts_filename), parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print ('\n Parsed Data:')

###############################################################################

print(data.head())
print('\n Data Types:')
print(data.dtypes)
data.index

###############################################################################
ts = data['#Passengers']
ts.head(10)

###############################################################################

plt.plot(ts)

###############################################################################

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd =pd.Series(timeseries).rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    

test_stationarity(ts)    

###############################################################################

ts_log = np.log(ts)
plt.plot(ts_log)


moving_avg = pd.Series(ts_log).rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

###############################################################################
expwighted_avg =  ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


###############################################################################

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

###############################################################################

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

###############################################################################

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

###############################################################################
