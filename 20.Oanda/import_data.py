# インポート
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline

#自分のアカウント、トークンをセット
accountID = "101-009-12923052-001"
access_token = '7b8e8782a87ce72d52cd23b2d74772df-13011574ae44dc0db94fc8dac52e4ebc'
api = API(access_token=access_token, environment="practice")

# 5分間隔で5000データ
params = {
    "count": 5000,
    "granularity": "M5"
}

# APIへ過去データをリクエスト
r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
api.request(r)

# dataとしてリストへ変換
data = []
for raw in r.response['candles']:
    data.append([raw['time'], raw['volume'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])

# リストからデータフレームへ変換
df = pd.DataFrame(data)
df.columns = ['time', 'volume', 'open', 'high', 'low', 'close']
df = df.set_index('time')

# date型を綺麗にする
df.index = pd.to_datetime(df.index)
print(df.tail())
df=df.astype(float)
#df['close'].plot(figsize=(10,5), linewidth=0.5)

# 一つ前の終値と現在の終値の差分
df['Close_Diff'] = df['close'] - df['close'].shift(1)

# 正規化する
df = pd.DataFrame((df['Close_Diff']-df['Close_Diff'].mean()) / (df['Close_Diff'].max() - df['Close_Diff'].min()), columns=['Close_Diff'])
df['Close_Diff'].plot()
