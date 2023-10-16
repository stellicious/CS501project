#!/usr/bin/env python
# coding: utf-8

# 

# In[6]:


from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import GRU

from keras.callbacks import History

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



df = pd.read_excel("C:/Users/panto/Desktop/CS-501Project/monthly_mean_anom.xlsx", index_col=0, na_values=['NA'], usecols = "N,O")
df.keys()

mean_anom = df.iloc[:, 0].values;
dates = df.index[:]

#total_tr = 1315

tr1 = mean_anom[0:279]
tr2 = mean_anom[609:1644]
training_data = np.concatenate((tr1,tr2))
test_data = mean_anom[280:608]

#training_data = mean_anom[0:1123] #80%
#test_data = mean_anom[1124:1403]

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))
x_training_data = []

y_training_data =[]


for i in range(1, len(training_data)):

    x_training_data.append(training_data[i-1:i, 0])

    y_training_data.append(training_data[i, 0])

    
    
x_training_data = np.array(x_training_data)

y_training_data = np.array(y_training_data)

x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], x_training_data.shape[1], 1))

rnn = Sequential()

#Adding our first LSTM layer

rnn.add(LSTM(units = 70, return_sequences = False,  activation=None, input_shape = (x_training_data.shape[1], 1)))

#Perform some dropout regularization

rnn.add(Dropout(0.1))

rnn.add(Dense(units = 1))

#boundaries = [100, 100]
#values = [0.01, 0.005, 0.001]
#learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

#lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-2,
#    decay_steps=200,
#    decay_rate=0.2)
#optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
#optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn)

opt = keras.optimizers.Adam(learning_rate=0.0001)

rnn.compile(optimizer=opt, loss = 'mean_squared_error')

rnn_history = rnn.fit(x_training_data, y_training_data, validation_split=0.5 ,epochs = 60, batch_size = 2,shuffle=False, verbose=1)

x_test_data = mean_anom[279:608]
#x_test_data = mean_anom[1123:1403]
x_test_data = np.reshape(x_test_data, (-1, 1))

x_test_data = scaler.transform(x_test_data)

final_x_test_data = []

for i in range(1, len(x_test_data)):

    final_x_test_data.append(x_test_data[i-1:i, 0])

final_x_test_data = np.array(final_x_test_data)

final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

predictions = rnn.predict(final_x_test_data)

unscaled_predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(15,7))
plt.plot(dates[280:608],unscaled_predictions, color = 'blue', label = "Predictions")

plt.plot(dates[280:608], test_data, color = 'red', label = "Real Data")



#plt.title('Valve Temperature Predictions')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Temperature(°C)')


import math
import statistics
from sklearn.metrics import mean_squared_error
#rmse = math.sqrt(statistics.mean((unscaled_predictions-test_data)**2));
#rmse=math.sqrt(mean_squared_error(test_data, unscaled_predictions))

# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(rnn_history.history['loss']), 'r', label='train')
ax.plot(np.sqrt(rnn_history.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

#rmse
#max_error(test_data, unscaled_predictions)


# In[ ]:





# In[7]:


from scipy.stats import norm
from matplotlib import pyplot
from pandas import DataFrame
residuals = [test_data[i]-unscaled_predictions[i] for i in range(len(unscaled_predictions))]
residuals = DataFrame(residuals)
# histogram plot
#residuals.hist()
plt.hist(residuals)
mu, std = norm.fit(residuals) 
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax)
p = norm.pdf(x, mu, std)

#plt.plot(x, p, 'k', linewidth=2)


plt.xlabel('Residual Error (°C)')
plt.ylabel('Number of Testing Points')
plt.title('Number of Testing Points vs. Residual Error')

pyplot.show()


# In[ ]:




