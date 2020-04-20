'''
サポートベクトル回帰と特徴選択
https://qiita.com/hrs1985/items/ba24fde9981f611cc7d8
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm

PI = 3.14

# 0～2πまでを120等分した点を作る
X = np.array(range(120))
X = X * 6 * PI / 360
# y=sinXを計算し、ガウス分布に従う誤差を加える
y = np.sin(X)
e = [random.gauss(0, 0.2) for i in range(len(y))]
y += e
# 列ベクトルに変換する
X = X[:, np.newaxis]

# 学習を行う
svr = svm.SVR(kernel='rbf')
svr.fit(X, y)

# 回帰曲線を描く
X_plot = np.linspace(0, 2*PI, 10000)
y_plot = svr.predict(X_plot[:, np.newaxis])

#グラフにプロットする。
plt.scatter(X, y)
plt.plot(X_plot, y_plot)
plt.show()