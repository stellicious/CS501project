#!/usr/bin/env python
# coding: utf-8

# In[11]:


from pandas import read_csv
from pandas import DataFrame
from matplotlib import pyplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import xlrd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import max_error

from statsmodels.tsa.arima.model import ARIMA

df = pd.read_excel("C:/Users/panto/Desktop/CS-501Project/monthly_mean_anom.xlsx", index_col=0, na_values=['NA'], usecols = "N,O")
df.keys()

mean_anom = df.iloc[:, 0].values;
dates = df.index[:]

tr1 = mean_anom[0:279]
tr2 = mean_anom[609:1644]
train_data = list(np.concatenate((tr1,tr2)))
test_data = list(mean_anom[280:608])

import time
from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
model_predictions = []
total_runtime = 0
for i in range(len(test_data)):
    model = SimpleExpSmoothing(train_data)
    model_fit = model.fit(0.9)
    start_time = time.time()
    output = model_fit.predict()
    prediction_time = time.time() - start_time
    total_runtime += prediction_time  # Add current prediction time to total runtime
    print("--- %s seconds ---" % prediction_time)
    yhat=list(output)[0]
    model_predictions.append(yhat)
    actual_test_value = test_data[i]
    train_data.append(actual_test_value)
    

average_runtime_per_point = total_runtime / len(test_data)
print("Average prediction runtime per testing point: %s seconds" % average_runtime_per_point)

plt.figure(figsize=(12,7))
date_range = dates[280:608]
plt.plot(date_range,model_predictions,color='blue',label='Predicted')
plt.plot(date_range,test_data,color='red',label='Actual')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Mean')

from scipy.stats import norm
from matplotlib import pyplot
residuals = [test_data[i]-model_predictions[i] for i in range(len(model_predictions))]
residuals = DataFrame(residuals)
plt.hist(residuals)
mu, std = norm.fit(residuals) 
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax)
p = norm.pdf(x, mu, std)

plt.xlabel('Residual Error (°C)')
plt.ylabel('Number of Testing Points')
plt.title('Number of Testing Points vs. Residual Error')

pyplot.show()

