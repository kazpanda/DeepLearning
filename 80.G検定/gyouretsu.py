# ライブラリー
import numpy as np

# 行列式
list1 = np.array([[1, -1], [2, 2]])　# 行毎に書く
list2 = np.array([[1, 1,1], [-1, 0,1]]) # 行毎に書く

# 行列積
result = np.dot(list1, list2)

print(result)
