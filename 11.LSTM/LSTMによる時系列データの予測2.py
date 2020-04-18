# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:18:22 2020

@author: kazpanda
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
#%matplotlib inline
sns.set()

df = pd.read_csv('AirPassengers.csv', index_col='Month', dtype={1: 'float'})
ts = df['#Passengers']
ts.head()

input_data = [] # train
output_data = [] # test (answer)
for i in range(0, 72):
    tmpX = []
    for j in range(0, 24):
        tmpX.append(ts[i+j])
    input_data.append(tmpX)
    
    tmpY = []
    for j in range(0, 12):
        tmpY.append(ts[24+i+j])
    output_data.append(tmpY)
    

np_input_data = np.array(input_data)
np_output_data = np.array(output_data)

train_X = np_input_data.reshape((np_input_data.shape[0], np_input_data.shape[1], 1))
train_Y = np_output_data.reshape((np_output_data.shape[0], np_output_data.shape[1], 1))


m = Sequential()
# 入力データ数が24なので、input_shapeの値が(24,1)です。
m.add(LSTM(100, activation='relu', input_shape=(24, 1)))
# 予測範囲は12ステップなので、RepeatVectoorに12を指定する必要があります。
m.add(RepeatVector(12)) 
m.add(LSTM(100, activation='relu', return_sequences=True))
m.add(TimeDistributed(Dense(1)))
m.compile(optimizer='adam', loss='mse')
m.fit(train_X, train_Y, epochs=1000, verbose=0)


# データ60番～83番から、次の一年(84番～95番)を予測
input = np.array(ts[60:84]) 
input = input.reshape((1,24,1))
yhat = m.predict(input)
 
# 可視化用に、予測結果yhatを、配列predictに格納
predict = []
for i in range(0, 12):
    predict.append(yhat[0][i])
 
# 比較するために実データをプロット
plt.plot(ts)
 
# 予測したデータをプロット
xdata = np.arange(84, 96, 1)
plt.plot(xdata, predict, 'r')


input = np.array(ts[72:96])
input = input.reshape((1,24,1))
yhat = m.predict(input)
 
predict = []
for i in range(0, 12):
    predict.append(yhat[0][i])
 
plt.plot(ts)
xdata = np.arange(96, 108, 1)
plt.plot(xdata, predict, 'r')

