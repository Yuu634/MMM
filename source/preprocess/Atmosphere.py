import os
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib
import numpy as np
import math
from sklearn import cluster
import pandas as pd
#import seaborn as sns
import IPython.display
import datetime
import platform
from midi2audio import FluidSynth
from pydub import AudioSegment
from scipy.cluster.hierarchy import dendrogram, linkage
import ast
#%matplotlib inline

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

#def pick_colors(df):
#   return  list(clrs.cnames.values())[:len(df.columns)]

def show_stackplot(index_df,df_list,colors):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    fig.patch.set_facecolor('white')
    ax.stackplot(index_df.index,df_list,colors=colors)
    plt.xticks([i*10 for i in range(int(round(index_df.index.tolist()[-1]))//10 + 1)])
    plt.show()

# ディレクトリ内の全MIDIファイルをWAVに変換し、リストに追加する関数
def convert_all_midi_to_wav(directory):
    fs = FluidSynth("Chaos_V20.sf2")
    
    for filename in os.listdir(directory):
        if filename.endswith('.mid') or filename.endswith('.midi'):
            wav_path = filename.replace('.mid', '.wav').replace('.midi', '.wav')
            if not (wav_path in os.listdir(directory)):
                midi_path = os.path.join(directory, filename)
                wav_path = os.path.join(directory, filename.replace('.mid', '.wav').replace('.midi', '.wav'))
                fs.midi_to_audio(midi_path, wav_path)

# 複数のWAVファイルを一つに結合する関数
def combine_wav_files(wav_files, output_path):
    combined = AudioSegment.empty()
    for wav_file in wav_files:
        sound = AudioSegment.from_wav(wav_file)
        combined += sound
    combined.export(output_path, format="wav")


""" メイン関数（全曲によるクラスタリング） """
def get_Atmosphere(source_path):
    """変数設定"""
    # フーリエ変換の初期設定(調整必要)
    n_fft = 1024 # データの取得幅
    hop_length = n_fft // 4 # 次の取得までの幅
    # クラスタ分類の数⇒次元数　(調整必要)
    n_clusters = 30
    # 学習データ全体でのクラスタ数
    all_clusters = 40
    # k_meansのラベルをいくつずつ見ていくか(調整必要)
    hop = 50
    # クラスタ数⇒Atmosphereの種類(調整必要)
    music_cluster_num = 4
    #クラスタくっつける個数（調整必要）
    min_num = 4
    
    # MIDIファイルが保存されているディレクトリのパスを指定
    ssh_env_vars = ['SSH_CONNECTION', 'SSH_CLIENT', 'SSH_TTY']
    #SSH接続時
    IsSSH = False
    for var in ssh_env_vars:
        if os.getenv(var):
            IsSSH = True
            directory = "/mnt/aoni04/obara/MMM-JSB/" + source_path
            break
    
    if IsSSH == False:
        if platform.system() == "Windows":
            directory = "C:/Users/yuki-/.vscode/Python/research/MMM-JSB/" + source_path 
        elif platform.system() == "Linux":
            directory = "/mnt/c/Users/yuki-/.vscode/Python/research/MMM-JSB/" + source_path
        """wavファイル取得"""
        convert_all_midi_to_wav(directory)
    
    #結果格納リスト
    Atmosphere_list = []
    columns = list(range(n_clusters)) + ['cluster', 'cluster_index','filename','index_time']
    cluster_mean_list = pd.DataFrame(columns=columns)
    for filename in os.listdir(directory):
        """ファイル拡張子確認"""
        #music:音楽情報のベクトル化 fs:サンプリングレート
        if filename.endswith('.wav'):
            wavfile = os.path.join(directory, filename)
            music,fs = librosa.load(wavfile)
        elif filename.endswith('mp3'):
            mp3file = os.path.join(directory, filename)
            music,fs = librosa.load(mp3file)
        else:
            continue
        
        """スペクトログラム取得"""
        D = librosa.stft(music,n_fft=n_fft,hop_length=hop_length,win_length=None)
        spectrogram = np.abs(D)**2
        
        """瞬間瞬間での雰囲気取得"""
        # クラスタ分類(k_means)
        logamp = librosa.amplitude_to_db(spectrogram,ref=np.max)
        k_means = cluster.KMeans(n_clusters=n_clusters)
        try:
            k_means.fit(logamp.T)
        except:
            print("Cluster Error")
            continue
        
        #kmeansのラベルごとにdbを
        col = k_means.labels_.shape[0]
        # グラフ作成
        count_list = []

        # ラベル付けたデータをhop分持ってきて数を数える。
        for i in range(col//hop):
            x = k_means.labels_[i*hop:(i+1)*hop]
            count = np.bincount(x)
            count_list.append(count)
        # index内容
        index = [(len(music)/fs)/len(count_list)*x for x in range(len(count_list))] # 秒数
        #hopごとに、どのクラスタが何回出てきているかをテーブル化
        df = pd.DataFrame(count_list,index = index).fillna(0)

        columns = [chr(i) for i in range(65,65+26)][:10]
        
        """Atmosphereトークン取得用クラスタリング"""
        #hopごとにクラスタ作成（約１秒に１つ）
        try:
            k_means_music = cluster.KMeans(n_clusters=music_cluster_num, n_init='auto')
            k_means_music.fit(df)
            #Atmosphere2 = cluster.KMeans(n_clusters=music_cluster_num*2, n_init='auto')
            #Atmosphere2.fit(df)
        except:
            print("Cluster Error")
            continue
        df['cluster']  = k_means_music.labels_
        """短いクラスタの結合"""
        comp_list = []
        m=1

        for i in range(len(k_means_music.labels_) - 1):
            if k_means_music.labels_[i] == k_means_music.labels_[i+1]:
                m = m + 1
            else:
                comp_list.append([k_means_music.labels_[i],m])
                m=1

        # 最後の文字をリストにくっつける。
        comp_list.append([k_means_music.labels_[-1],m])

        # comp_listの長さが短いものは、前のクラスタと同じIDにする。
        for i in range(1,min_num):
            replace_comp_list = []
            replace_comp_list.append(comp_list[0])

            for j in range(1,len(comp_list)):
                if comp_list[j][1] == i:
                    replace_comp_list[-1][1] += i
                else:
                    replace_comp_list.append(comp_list[j])

            # 同じクラスタが並んでる場合はくっつける
            k = 0
            while k < len(replace_comp_list)-1:
                try:
                    while replace_comp_list[k][0] == replace_comp_list[k+1][0]:          
                        replace_comp_list[k][1] += replace_comp_list[k+1][1]
                        replace_comp_list.pop(k+1)
                except IndexError:
                    continue
                else:
                    k += 1

            comp_list = replace_comp_list
        # 元の形式に戻す
        thawing_list = []
        for i in range(len(replace_comp_list)):
            for j in range(replace_comp_list[i][1]):
                thawing_list.append(replace_comp_list[i][0])
        
        """各クラスタごとのdfの平均値を計算"""
        df['cluster'] = thawing_list
        thawing_mean = df.groupby('cluster').mean()
        thawing_mean['cluster'] = thawing_mean.index
        # clusterごとのindexリストをカラムに追加
        cluster_indices = df.groupby('cluster').apply(lambda x: x.index.tolist())
        thawing_mean['cluster_index'] = thawing_mean['cluster'].map(cluster_indices)
        #全曲クラスタリング用データに追加
        thawing_mean['filename'] = filename
        thawing_mean['index_time'] = (len(music)/fs)/len(count_list)
        
        cluster_mean_list = pd.concat([cluster_mean_list.reset_index(drop=True), thawing_mean.reset_index(drop=True)])
    
    #欠損値（Nan）の補完
    """for column_name in range(n_clusters):
        column_name = str(column_name)
        cluster_mean_list[column_name] = cluster_mean_list.groupby("filename")[column_name].transform(
            lambda group: group.fillna(group.mean() if not pd.isna(group.mean()) else 0)
        )"""
    target_columns = df.columns[:df.columns.get_loc("cluster")]
    for column_name in target_columns:
        cluster_mean_list[column_name] = cluster_mean_list[column_name].fillna(cluster_mean_list[column_name].mean())
    #Excelに保存
    #cluster_mean_list.to_csv("Atmosphere_label.csv", index=False, encoding="utf-8")
    #全曲のcluster中心点でKmeans
    allmusic_kmeans = cluster.KMeans(n_clusters=all_clusters, n_init='auto')
    allmusic_kmeans.fit(cluster_mean_list.loc[:, cluster_mean_list.columns[:cluster_mean_list.columns.get_loc('cluster')]])
    cluster_list = pd.DataFrame()
    cluster_list = cluster_list.reset_index(drop=True)
    cluster_mean_list = cluster_mean_list.reset_index(drop=True)
    cluster_list['cluster'] = allmusic_kmeans.labels_.tolist()
    cluster_list = pd.concat([cluster_list, cluster_mean_list.loc[:, 'cluster_index':]], axis=1)

    """Atmosphere辞書格納(Atmosphereのみ)"""
    for row in cluster_list.itertuples():
        element = {}
        for time in row.cluster_index:
            element[time] = (row.cluster,None)
        found = False
        
        for entry in Atmosphere_list:
            if row.filename in entry:
                # すでに存在する場合、その value に element を加える
                entry[row.filename].update(element)
                found = True
                break 
        # 存在しない場合、新しいエントリを追加
        if not found:
            Atmosphere_list.append({row.filename:element})
    
    return Atmosphere_list

"""保存済みクラスタがある場合"""
def csv_to_Atmosphere(filename):
    cluster_mean_list = pd.read_csv(filename)
    n_clusters=30
    all_clusters=40
    #index 文字列⇒リストに変換
    cluster_mean_list["cluster_index"] = cluster_mean_list["cluster_index"].apply(ast.literal_eval)
    #欠損値（Nan）の補完
    for column_name in range(n_clusters):
        column_name = str(column_name)
        cluster_mean_list[column_name] = cluster_mean_list.groupby("filename")[column_name].transform(
            lambda group: group.fillna(group.mean() if not pd.isna(group.mean()) else 0)
        )
    #cluster_mean_list.to_csv("Atmosphere_label.csv", index=False, encoding="utf-8")
    #全曲のcluster中心点でKmeans
    allmusic_kmeans = cluster.KMeans(n_clusters=all_clusters, n_init='auto')
    allmusic_kmeans.fit(cluster_mean_list.loc[:, cluster_mean_list.columns[:cluster_mean_list.columns.get_loc('cluster')]])
    cluster_list = pd.DataFrame()
    cluster_list = cluster_list.reset_index(drop=True)
    cluster_mean_list = cluster_mean_list.reset_index(drop=True)
    cluster_list['cluster'] = allmusic_kmeans.labels_.tolist()
    cluster_list = pd.concat([cluster_list, cluster_mean_list.loc[:, 'cluster_index':]], axis=1)

    """Atmosphere辞書格納(Atmosphereのみ)"""
    Atmosphere_list = []
    for row in cluster_list.itertuples():
        element = {}
        for time in row.cluster_index:
            element[time] = (row.cluster,None)
        found = False
        
        for entry in Atmosphere_list:
            if row.filename in entry:
                # すでに存在する場合、その value に element を加える
                entry[row.filename].update(element)
                found = True
                break
        # 存在しない場合、新しいエントリを追加
        if not found:
            Atmosphere_list.append({row.filename:element})
    
    return Atmosphere_list
    
#get_Atmosphere("testDataset")
#csv_to_Atmosphere("Atmosphere_label.csv")