import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset


# ターミナル対応
import matplotlib
matplotlib.use('Agg')
 
def load_tsdata():
    # csvからデータの読み込み
 
    # 読み込むファルのリスト
    busyo_l = ['busyo1.csv', 'busyo2.csv', 'busyo3.csv']
    #busyo_l = ['../00.data/busyo1.csv', '../00.data/busyo2.csv', '../00.data/busyo3.csv']
 
    df = pd.DataFrame()
    for i, busyo_csv in enumerate(busyo_l):
        # はじめの2行はスキップ
        # 1列目（週）をインデックスに指定し、datetime型で読み込み
        # tmp_df = pd.read_csv(busyo_csv, skiprows=2, parse_dates=True, index_col='week')
        tmp_df = pd.read_csv(busyo_csv, parse_dates=True, index_col='週')
 
        # カラム名から「: (日本)」を除去
        #names = tmp_df.columns.tolist()
        #tmp_df.columns = [x.replace(': (日本)', '') for x in names]
 
        # 最初のファイル以外では「織田信長」を除外
        if i != 0:
            tmp_df.drop(columns=['織田信長'], inplace=True)
 
        df = pd.concat([df, tmp_df], axis=1)
 
    print(df.index.dtype)
    print(df.dtypes)
 
    # 読み込んだデータのプロット
    plt.figure()
    df.plot()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('busho.png')
    plt.clf()
 
    return df

def tsclusteringN(ts_data, names):
    # クラスタリング
 
    # 正規化
    ts_dataset = TimeSeriesScalerMinMax().fit_transform(ts_data)
 
    metric = 'dtw'
    n_clusters = [n for n in range(2, 6)]
    for n in n_clusters:
        print('クラスター数 =', n)
 
        # metricが「DTW」か「softdtw」なら異なるデータ数の時系列データでもOK
        km = TimeSeriesKMeans(n_clusters=n, metric=metric, verbose=False, random_state=1).fit(ts_dataset)
 
        # クラスタリングの結果
        print('クラスタリング結果 =', km.labels_)
 
        # -1から1の範囲の値。シルエット値が1に近く、かつシルエット値をプロットしたシルエット図でクラスター間の幅の差が最も少ないクラスター数が最適
        # 今回はシルエット値のみを確認
        print('シルエット値 =', silhouette_score(ts_dataset, km.labels_, metric=metric))
        print()

def plot_clustering(km, ts_dataset, names, n_clusters):
    # クラスタリングの結果をプロット
 
    # クラスターごとの中心をプロット
    for i, c in enumerate(km.cluster_centers_):
        plt.plot(c.T[0], label=i)
 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    plt.savefig('ts_clust_center.png')
    plt.clf()
 
    # クラスターごとのプロット
    for i in range(n_clusters):
        for label, d, t in zip(km.labels_, ts_dataset, names):
            if label == i:
                plt.plot(d, label=t)
 
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.savefig('ts_labeled{}.png'.format(i))
        plt.clf()
 
def tsclustering(ts_data, names):
    # 正規化
    ts_dataset = TimeSeriesScalerMinMax().fit_transform(ts_data)
 
    n_clusters = 2
    metric = 'dtw'
 
    # metricが「DTW」か「softdtw」なら異なるデータ数の時系列データでもOK
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, verbose=False, random_state=1).fit(ts_dataset)
 
    # クラスタリングの結果
    print('クラスタリング結果 =', km.labels_)
 
    plot_clustering(km, ts_dataset, names, n_clusters)


# メイン処理  
def main():
    df = load_tsdata()
     #tsclusteringN(df.values.transpose(), df.columns)
    tsclustering(df.values.transpose(), df.columns)

if __name__ == '__main__':
    main()