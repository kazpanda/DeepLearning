import random
import pandas as pd
import time 
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
plt.style.use('ggplot')

bitcoin_market_info = pd.read_csv('../00.data/GBPJPY_day_api.csv')
#bitcoin_market_info = pd.read_csv('GBPJPY_day_api.csv', index_col='day')
#df.index = pd.to_datetime(df.index)
#bitcoin_market_info = bitcoin_market_info.drop(["day","time"],axis=1)
bitcoin_market_info = bitcoin_market_info.drop(["time"],axis=1)
bitcoin_market_info =bitcoin_market_info.reset_index()
datasize=bitcoin_market_info.shape[0]

#bitcoin_market_info.head()

#bitcoin_market_info.info()



# データの前処理(1)

wide=60#何日前のデータまで見るか

#連続値を離散値にする関数（閾値は変化率0.01)
f=lambda x: 2 if x>0.01 else 0 if x<-0.01 else 1 if -0.01<=x<=0.01 else np.nan

def seikei(df):
    random.shuffle([i for i in range(datasize-wide-2)])#RNNでは学習する順番によっても結果が変わってくるので、順番をバラバラにできるよう準備しておきます
    shuffle_index = []
    test_index=[]
    train_index=[]

#    shuffle_index = range(:datasize//3)
#    shuffle_index = range(datasize//3:)
    n_samples = datasize # データの個数
    n_train   = n_samples // 2 # 半分のデータを学習
    n_test    = n_samples - n_train # テストデータ数
 
    test_index = range(0, n_train)
    train_index = range(n_train, n_samples)

#    test_index=shuffle_index[:datasize//3]
#    train_index=shuffle_index[datasize//3:]
    
    df_train_list=[]
    df_test_list=[]
    df_list=[]
    keys=["{}".format(i) for i in range(wide)]
    columns=df.columns
    
    #正解ラベルの作成
    #close_diff=df.loc[:,"Close**"].pct_change(-1).map(f).rename(columns={'Close**': 'diff'})[0:datasize-wide-2]
    #close_diff=df.loc[:,"close"].pct_change(-1).map(f).rename(columns={'close': 'diff'})[0:datasize-wide-2]
    close_diff=df.loc[:,"close"]

    y_train=close_diff[train_index]
    y_test=close_diff[test_index]
    
    diff_list=[]
    #変分からなるデータフレームに書き換える
    for col in columns:
        data=df.loc[:,col]
        diff_data_cleaned=preprocessing.scale(data.pct_change(-1)[:datasize-1]) #価格変動をみたいので差分を取り、精度を上げるために標準化しています。
        diff_data_cleaned.index=range(datasize-1)
        diff_list.append(pd.Series(data=diff_data_cleaned, dtype='float'))
        
    df=pd.concat(diff_list,axis=1)

    for column in columns:
        series_list=[df.loc[:,column]]
        for i in range(wide):
            series_kari=series_list[0].drop(0)
            series_kari.index=range(datasize-(i+2))
            series_list.insert(0,series_kari)
            
        concat_df=pd.concat(series_list,axis=1,keys=keys).drop(0).dropna()
        concat_df.index=range(datasize-(wide+2))
        
        concat_df_train=concat_df.iloc[train_index,:]
        concat_df_test=concat_df.iloc[test_index,:]
        
        df_train_list.append(concat_df_train)
        df_test_list.append(concat_df_test)
    return df_train_list,df_test_list,y_train,y_test



# データの前処理(2)

def convert_threeDarray_for_nn(df_list):
    array_list = []
    for df in df_list:
        ndarray = np.array(df)
        array_list.append(np.reshape(
            ndarray, (ndarray.shape[0], ndarray.shape[1], 1)))

    return np.concatenate(array_list, axis=2)

# データの前処理(3)
#train_df_list,test_df_list,Y_train,Y_test=seikei(df)
train_df_list,test_df_list,Y_train,Y_test=seikei(bitcoin_market_info)

X_train = convert_threeDarray_for_nn(train_df_list)
X_test = convert_threeDarray_for_nn(test_df_list)

n_classes =3
Y_train = to_categorical(Y_train, n_classes)
Y_test = to_categorical(Y_test, n_classes)

input_size = [X_train.shape[1], X_train.shape[2]]#入力するデータサイズを取得


# 指標の設定
def to_array(y):
  array=[]
  for i in range(y.shape[0]):
    array.append(y[i].argmax())
  return(array)

def kentei(predict_y,test_y):
  count=0
  for i in range(len(predict_y)):
    if predict_y[i]==2 and test_y[i]==0:
      count+=1
  return count/predict_y.count(2)

# モデルの作成
def pred_activity_lstm(input_dim,
                       activate_method='softmax',  # 活性化関数
                       loss_method='categorical_crossentropy',  # 損失関数
                       optimizer_method='adam',  # パラメータの更新方法
                       kernel_init_method='glorot_normal',  # 重みの初期化方法
                       batch_normalization=False,  # バッチ正規化
                       dropout_rate=None  # ドロップアウト率
                       ):
    
    model = Sequential()
    model.add(
        LSTM(
            input_shape=(input_dim[0], input_dim[1]),
            units=60,
            kernel_initializer=kernel_init_method,
            return_sequences=True
        ))

    if batch_normalization:
        model.add(BatchNormalization())

    if dropout_rate:
        model.add(Dropout(dropout_rate))

    model.add(
        LSTM(
            units=30,
            kernel_initializer=kernel_init_method,
            return_sequences=False 
        ))

    if batch_normalization:
        model.add(BatchNormalization())

    if dropout_rate:
        model.add(Dropout(dropout_rate))

    model.add(Dense(units=n_classes, activation=activate_method))
    model.compile(loss=loss_method, optimizer=optimizer_method,
                  metrics=['accuracy'])

    return model

turned_model = pred_activity_lstm(
    input_dim=input_size,
    activate_method='softmax',
    loss_method='categorical_crossentropy',
    optimizer_method='adam',
    kernel_init_method='glorot_normal',
    batch_normalization=True
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


# 学習
# 学習スタート
history = turned_model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.3,
    callbacks=[early_stopping],
    verbose=2
)

score = lstm_model.evaluate(X_test, Y_test, verbose=1)


# 精度の推移図を出力
plt.figure(figsize=(8, 5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 損失関数の推移図を出力
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = lstm_model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_y = to_array(Y_test)
pred_y = to_array(turend_model.predict(X_test))

print("投資失敗率:{}".format(kentei(pred_y,test_y)))
