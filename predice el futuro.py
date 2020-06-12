# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:05:28 2020

@author: windows 10
"""

from google.colab import drive
drive.mount('/content/drive/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("/content/drive/My Drive/train_csv.csv")
test=pd.read_csv("/content/drive/My Drive/test_csv.csv")
train.head()
test.head()
train.info()
train.isnull().sum()
#Converting time object to datetime
train['time']= pd.to_datetime(train['time']) 
train.info()
train.tail(10)
train.describe()
train.time.value_counts()
train.feature.value_counts()
train=train.drop(columns='id',axis=1)
train.set_index('time',inplace=True)
train.index
from statsmodels.tsa.stattools import adfuller

test_result=adfuller(train['feature'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(feature):
    result=adfuller(feature)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
        

adfuller_test(train['feature'])
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(train['feature'])
plt.show()
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(train, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(train, ax=pyplot.gca())
pyplot.show()
#Parameter Selection for the ARIMA Time Series Model
import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train['feature'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
import statsmodels.api as sm

model=sm.tsa.statespace.SARIMAX(train['feature'],order=(1, 1, 1),seasonal_order=(0,1,1,12),enforce_stationarity=False,enforce_invertibility=False)
results=model.fit()
results.summary()
results.plot_diagnostics(figsize=(16, 8))
plt.show()

train['forecast']=results.predict(start=pd.to_datetime('2019-03-19 00:10:00'),dynamic=True)
train[['feature','forecast']].plot(figsize=(12,8))
y_forecasted = train['forecast']
y_truth = train['feature']['2019-03-19 00:10:00':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=40)
pred_uc
pred_ci = pred_uc.conf_int()
pred_ci 
future_datest_df=pd.DataFrame(index=pred_ci.index,columns=train.columns)
print(future_datest_df)
future_df=pd.concat([train,future_datest_df])
print(future_df.tail())
future_df['forecast'] = results.predict(start ='2019-03-19 00:13:00',end=120 ,dynamic= True)  
future_df[['feature', 'forecast']].plot(figsize=(12, 8))
ytest= pd.DataFrame(pred_uc.predicted_mean)
ytest = ytest.reset_index(drop=True)
print(ytest)
#now taking test data
#Converting time object to datetime
test['time']= pd.to_datetime(test['time'])
test.head()
#Adding two Dataframes
test_pred = pd.concat([test, ytest], axis=1)
print(test_pred)
#Adding column name
test_pred['feature']=test_pred[0]
#column not required removing
test_pred=test_pred.drop(columns=['time',0],axis=1)
print(test_pred)
test_pred.to_csv('submission.csv',index=False)