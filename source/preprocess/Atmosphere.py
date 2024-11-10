import os
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib
import numpy as np
import math
from sklearn import cluster
import pandas as pd
import seaborn as sns
import IPython.display
import datetime
import platform
from midi2audio import FluidSynth
from pydub import AudioSegment
from scipy.cluster.hierarchy import dendrogram, linkage
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

def pick_colors(df):
    return  list(clrs.cnames.values())[:len(df.columns)]

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


""" メイン関数（曲ごとにクラスタリング） """
def get_Atmosphere(source_path):
    """変数設定"""
    # フーリエ変換の初期設定(調整必要)
    n_fft = 1024 # データの取得幅
    hop_length = n_fft // 4 # 次の取得までの幅
    # クラスタ分類の数(調整必要)
    n_clusters=20
    # k_meansのラベルをいくつずつ見ていくか(調整必要)
    hop = 50
    # クラスタ数⇒Atmosphereの種類(調整必要)
    music_cluster_num = 4
    #クラスタくっつける個数（調整必要）
    min_num = 5
    
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

    for filename in os.listdir(directory):
        """ファイル拡張子確認"""
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
        k_means.fit(logamp.T)
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
        df = pd.DataFrame(count_list,index = index).fillna(0)

        columns = [chr(i) for i in range(65,65+26)][:10]

        df_list = df_to_df_list(df)
        colors = pick_colors(df)
        #show_stackplot(df,df_list,colors)

        """階層的クラスタリング
        plt.figure(figsize=(10, 7))
        clu = linkage(df,method='ward')
        dendrogram(clu)
        plt.show()"""
        
        """Atmosphereトークン取得用クラスタリング"""
        k_means_music = cluster.KMeans(n_clusters=music_cluster_num, n_init='auto')
        k_means_music.fit(df)
        Atmosphere2 = cluster.KMeans(n_clusters=music_cluster_num*2, n_init='auto')
        Atmosphere2.fit(df)
        df['cluster']  = k_means_music.labels_

        df4 = labels_list_to_df(k_means_music.labels_)
        df4_list = df_to_df_list(df4)
        colors = pick_colors(df4)
        #show_stackplot(df,df4_list,colors)

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
                
        #print(thawing_list)

        """結果取得 （時間：クラスタ値）"""
        """result = {}
        result['filename'] = filename
        n=0
        result[0] = thawing_list[0]
        last_Atmoshere = thawing_list[0]
        for Atmosphere in thawing_list:
            if Atmosphere != last_Atmoshere:
                result[index[n]] = Atmosphere
                last_Atmoshere = Atmosphere
            n += 1
        Atmosphere_list.append(result)
        """
        
        """Atmosphereリスト格納
        Atmosphere_list = []
        element = []
        time = 0
        last_Atmosphere = thawing_list[0]
        element.append({index[0]:thawing_list[0]})
        for Atmosphere,Atmosphere2 in zip(thawing_list,Atmosphere2.labels_):
            #Atmosphereの切り替わるタイミングのみAtmosphere追加
            if Atmosphere != last_Atmosphere:
                element.append({index[time]:Atmosphere})
                last_Atmosphere = Atmosphere
            element.append({index[time]:Atmosphere2})
            time += 1
        Atmosphere_list.append({filename:element})
        """
        
        """Atmosphere辞書格納"""
        element = {}
        time = 0
        last_Atmosphere = None
        for atmos,atmos2 in zip(thawing_list,Atmosphere2.labels_):
            #Atmosphereの切り替わるタイミングのみAtmosphere追加
            if atmos != last_Atmosphere:
                element[index[time]] = (atmos,atmos2)
                last_Atmosphere = atmos
            else:
                element[index[time]] = (None,atmos2)
            time += 1
        Atmosphere_list.append({filename:element})
        df5 = labels_list_to_df(thawing_list)
        df5_list = df_to_df_list(df5)
        colors = pick_colors(df5)

        """結果の表示"""
        #show_stackplot(df,df5_list,colors)

        """
        plt.figure(figsize=(10, 6))
        plt.title('Power spectrogram')
        #plt.colorbar(format='%+2.0f dB')
        plt.tight_layout
        #show_stackplot(df,df4_list,colors)
        plt.show()"""
    
    return Atmosphere_list
    
#get_Atmosphere("test")