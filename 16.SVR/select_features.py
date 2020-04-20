def standardize(data_table):
    for column in data_table.columns:
        if column in ["target"]:
            continue
        if data_table[column].std() == 0:
            data_table.loc[:, column] = 0
        else:
            data_table.loc[:, column] = ((data_table.loc[:,column] - data_table[column].mean()) 
                                         / data_table[column].std())

    return data_table

# 感度の計算を行うメソッドです
def calculate_sensitivity(data_frame, feature_name, k=10):
    import numpy as np
    import pandas as pd
    from sklearn import svm
    from sklearn import linear_model
    from sklearn import grid_search

    # グリッドサーチ用のパラメータを設定します
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10**i for i in range(-4, 0)],
                         'C': [10**i for i in range(1,4)]}]

    # 傾きの値を格納するリストです。
    slope_list = []

    # サンプルサイズ
    sample_size = len(data_frame.index)

    features = list(data_frame.columns)
    features.remove("target")

    for number_set in range(k):

        # データを学習用とテスト用に分割します。
        if number_set < k - 1:
            test_data = data_frame.iloc[number_set*sample_size//k:(number_set+1)*sample_size//k,:]
            learn_data = pd.concat([data_frame.iloc[0:number_set*sample_size//k, :],data_frame.loc[(number_set+1)*sample_size//k:, :]])
        else:
            test_data = data_frame[(k-1)*sample_size//k:]
            learn_data = data_frame[:(k-1)*sample_size//k]
        # それぞれをラベルと特徴量に分割します
        learn_label_data = learn_data["target"]
        learn_feature_data = learn_data.loc[:,features]
        test_label_data = test_data["target"]
        test_feature_data = test_data.loc[:, features]

        # テストデータの感度分析を行う対象以外の列を、列平均で置換します。
        for column in test_feature_data.columns:
            if column == feature_name:
                continue
            test_feature_data.loc[:, column] = test_feature_data[column].mean()

        # SVRのために、それぞれのデータをnumpy.array形式に変換します。
        X_test = np.array(test_feature_data)
        X_linear_test = np.array(test_feature_data[feature_name])
        X_linear_test = X_linear_test[:, np.newaxis]
        y_test = np.array(test_label_data)
        X_learn = np.array(learn_feature_data)
        y_learn = np.array(learn_label_data)

        # 回帰分析を行い、出力を得ます
        gsvr = grid_search.GridSearchCV(svm.SVR(), tuned_parameters, cv=5, scoring="mean_squared_error") 
        gsvr.fit(X_learn, y_learn)
        y_predicted = gsvr.predict(X_test)

        # 出力に対する線形回帰を行います。
        lm = linear_model.LinearRegression()
        lm.fit(X_linear_test, y_predicted)

        # 傾きを取得します
        slope_list.append(lm.coef_[0])

    return np.array(slope_list).mean()

# 決定係数の計算を行うメソッドです。
def calculate_R2(data_frame,k=10):
    import numpy as np
    import pandas as pd
    from sklearn import svm
    from sklearn import grid_search

    # グリッドサーチ用のパラメータを設定します
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10**i for i in range(-4, 0)],
                         'C': [10**i for i in range(1,4)]}]
    svr = svm.SVR()

    # 各回の決定係数の値を格納するリストを定義します。
    R2_list = []

    features = list(data_frame.columns)
    features.remove("target")

    # サンプルサイズ
    sample_size = len(data_frame.index)

    for number_set in range(k):

        # データを学習用とテスト用に分割します。
        if number_set < k - 1:
            test_data = data_frame[number_set*sample_size//k:(number_set+1)*sample_size//k]
            learn_data = pd.concat([data_frame[0:number_set*sample_size//k],data_frame[(number_set+1)*sample_size//k:]])
        else:
            test_data = data_frame[(k-1)*sample_size//k:]
            learn_data = data_frame[:(k-1)*sample_size//k]
        # それぞれをラベルと特徴量に分割します
        learn_label_data = learn_data["target"]
        learn_feature_data = learn_data.loc[:, features]
        test_label_data = test_data["target"]
        test_feature_data = test_data.loc[:, features]

        # SVRのために、それぞれのデータをnumpy.array形式に変換します。
        X_test = np.array(test_feature_data)
        y_test = np.array(test_label_data)
        X_learn = np.array(learn_feature_data)
        y_learn = np.array(learn_label_data)

        # 回帰分析を行い、テストデータに対するR^2を計算します。
        gsvr = grid_search.GridSearchCV(svr, tuned_parameters, cv=5, scoring="mean_squared_error") 
        gsvr.fit(X_learn, y_learn)
        score = gsvr.best_estimator_.score(X_test, y_test)
        R2_list.append(score)

    # R^2の平均値を返します。
    return np.array(R2_list).mean()

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn import svm
    import pandas as pd
    import random
    import numpy as np

    # ボストンの家賃のデータを読み込む。
    boston = load_boston()
    X_data, y_data = boston.data, boston.target
    df = pd.DataFrame(X_data, columns=boston["feature_names"])
    df['target'] = y_data
    count = 0
    temp_data = standardize(df)
    # 交差検証のためにデータをランダムソートします。
    temp_data.reindex(np.random.permutation(temp_data.index)).reset_index(drop=True)
    # 各ループにおける特徴量の感度と決定係数を格納するためのdataframeを作ります。
    result_data_frame = pd.DataFrame(np.zeros((len(df.columns), len(df.columns))), columns=df.columns)
    result_data_frame["決定係数"] = np.zeros(len(df.columns))
    # 特徴量を削りきるまで以下のループを実行します。
    while(len(temp_data.columns)>1):
        # このラウンドにおける残った全特徴量を使用した場合の決定係数です。
        result_data_frame.loc[count, "決定係数"] = calculate_R2(temp_data,k=10)
        # このラウンドでの特徴量ごとの感度を格納するデータフレームです
        temp_features = list(temp_data.columns)
        temp_features.remove('target')
        temp_result = pd.DataFrame(np.zeros(len(temp_features)),
                                   columns=["abs_Sensitivity"], index=temp_features)

        # 各特徴量ごとに以下をループします。
        for i, feature in enumerate(temp_data.columns):
            if feature == "target":
                continue
            # 感度分析を行います。
            sensitivity = calculate_sensitivity(temp_data, feature)

            result_data_frame.loc[count, feature] = sensitivity
            temp_result.loc[feature, "abs_Sensitivity"] = abs(sensitivity)
            print(feature, sensitivity)

        print(count, result_data_frame.loc[count, "決定係数"])
        # 感度の絶対値が最小の特徴量をデータから除いたコピーを作ります。
        ineffective_feature = temp_result["abs_Sensitivity"].argmin()
        print(ineffective_feature)
        temp_data = temp_data.drop(ineffective_feature, axis=1)


    # データおよび、感度とR^2の推移を返します。
        result_data_frame.to_csv("result.csv")

        count += 1