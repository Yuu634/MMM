import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import math
from sklearn import cluster
import pandas as pd
import seaborn as sns
import IPython.display
import datetime
from scipy.cluster.hierarchy import dendrogram, linkage

# mp3の読み込み
music,fs = librosa.load('test/test.mp3')

# 後ほど用いる関数
def labels_list_to_df(labels_list):
    labels_01_list = []
    
    for i in range(len(labels_list)):
        labels_01_list.append(np.bincount([labels_list[i]]))

    return pd.DataFrame(labels_01_list)

def df_to_df_list(df):
    df_list =[]
    for i in range(len(df.columns)):
        df_list.append(df[i])
        
    return df_list

def pick_colors(df):
    return  list(clrs.cnames.values())[:len(df.columns)]

def show_stackplot(index_df,df_list,colors):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    fig.patch.set_facecolor('white')
    ax.stackplot(index_df.index,df_list,colors=colors)
    plt.xticks([i*10 for i in range(int(round(index_df.index.tolist()[-1]))//10 + 1)])
    plt.show()


# フーリエ変換の初期設定(調整必要)
n_fft = 2048 # データの取得幅
hop_length = n_fft // 4 # 次の取得までの幅
D = librosa.stft(music, n_fft=n_fft, hop_length=hop_length)

# メル周波数スペクトルを計算
mel_spectrogram = librosa.feature.melspectrogram(y=music, sr=fs, n_fft=n_fft, hop_length=hop_length)

# 平均値を取ることで特徴量を作成
features = np.mean(mel_spectrogram, axis=1)

# 階層的クラスタリング
Z = linkage(features.reshape(-1, 1), method='ward')

# デンドログラムの描画
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

from scipy.cluster.hierarchy import fcluster

# クラスタ数を決定（例：3）
max_d = 5
clusters = fcluster(Z, max_d, criterion='maxclust')

# 各クラスタのインデックスを取得
for cluster in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    print(f'Cluster {cluster}: Indices {cluster_indices}')

# 音楽の波形をプロット
plt.figure(figsize=(14, 5))
plt.plot(music)
plt.title('Waveform of Music')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()