# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ライブラリーのインポート
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
#%matplotlib inline
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D

# データの準備
timeline = np.arange(10000)
epochs = 10

# サインデータ生成関数
def sinnp(n, line):
    return np.sin(line * n / 100)

# コサインデータ生成関数
def cosnp(n, line):
    return np.cos(line * n / 100)

# サンプルデータの生成
raw_data = (sinnp(1, timeline) + sinnp(3, timeline) + sinnp(10, timeline) + cosnp(5, timeline) + cosnp(7, timeline)) / 5
# ノイズ処理
raw_data = raw_data + (np.random.rand(len(timeline)) * 0.1)# ノイズ項
# 描画
plt.plot(timeline[:1000], raw_data[:1000])
plt.xlabel("時間")
plt.ylabel("測定値")
plt.show()

input_data = []
output_data = []
for n in range(10000-80):
    input_data.append(raw_data[n:n+64])
    output_data.append(raw_data[n+64:n+80])

input_data = np.array(input_data)
output_data = np.array(output_data)
print(input_data.shape)
print(output_data.shape)

train_X = np.reshape(input_data, (-1, 64, 1))
train_Y = np.reshape(output_data, (-1, 16, 1))
print(train_X.shape)
print(train_Y.shape)

model = Sequential()
model.add(Conv1D(64, 8, padding='same', input_shape=(64, 1), activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(32, 8, padding='same', activation='relu'))
model.add(Conv1D(1, 8, padding='same', activation='tanh'))

model.compile(loss='mse', optimizer='adam')

model.summary()

history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epochs)

start = 9100
sheed = np.reshape(raw_data[start:start+64], (1, 64, 1))
prediction = sheed

for i in range(20):
    res = model.predict(sheed)
    sheed = np.concatenate((sheed[:, 16:, :], res), axis=1)
    prediction = np.concatenate((prediction, res), axis=1)
    

print(prediction.shape)
predictor = np.reshape(prediction, (-1))
print(predictor.shape)
plt.plot(range(len(predictor)), predictor, label='predict')
plt.plot(range(len(predictor)), raw_data[start:start + len(predictor)], label='real')
plt.legend() 
plt.show()

    