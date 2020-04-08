import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
import pandas as pd
from data2csv import save_csv, save_dataset
import math

# Prompt user for stock symbol
stock = input('Stock: ')
stock = stock.upper()
save_csv(stock, 'DAILY', '')

num_epochs = int(input('Number of Epochs for Training: '))

# Get dataset returned from save_dataset
X_train, X_test, Y_train, Y_test, scl, data = save_dataset(f'./charts/{stock}_daily.csv')

#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,Y_train,epochs=num_epochs,validation_data=(X_test,Y_test),shuffle=False)

import matplotlib.pyplot as plt

# Plot Loss vs Time Graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

# Plot prediction
Xt = model.predict(X_test)
#plt.plot(scl.inverse_transform(Y_test.reshape(-1,1)))
#plt.plot(scl.inverse_transform(Xt))
result = []
train_length = math.ceil(len(data)*.8)
valid = data[train_length:]
valid['Predictions'] = scl.inverse_transform(Xt)

for i in range(0, train_length):
    result.append(None)

for i in range(0, len(valid['Predictions']-1)):
    result.append(valid['Predictions'][i])

train = result[:train_length]
predicted = result[train_length:]
plt.title(stock + ' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Closing Price USD ($)')
# plt.plot(result[train_length:])
# plt.plot(result[:train_length])
plt.plot(data)
plt.plot(result)
plt.show()

from datetime import datetime
model.save(f'create_lstm_model.h5')

# print("Train: " + data)
# print("Result: " + result)


# i = 249
# Xt = model.predict(X_test[i].reshape(1,7,1))
# print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt),scl.inverse_transform(Y_test[i].reshape(-1,1))))
# predicted.append(scl.inverse_transform(Xt))
# actual.append(scl.inverse_transform(Y_test[i].reshape(-1,1)))

# result_df = pd.DataFrame({'predicted':list(np.reshape(predicted, (-1))),'actual':list(np.reshape(actual, (-1)))})

# Xt = model.predict(X_test)
# plt.plot(scl.inverse_transform(Y_test.reshape(-1,1)))
# plt.plot(scl.inverse_transform(Xt))
# plt.show()