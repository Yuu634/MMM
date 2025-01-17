from pathlib import Path
import os
import music21

import librosa
import librosa.display
import matplotlib.pyplot as plt
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from source import logging
import random
import numpy as np
import pandas as pd


#学習データの修正
#次Atmosphereトークンを最後に結合
#転調によるデータを別々に記述
def process_text_file(input_file, output_file, span):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    result = []
    start_lines = []
    #楽曲開始の行を取得
    for i in range(len(lines)):
        line = lines[i].strip()
        if "PIECE_START" in line:
            start_lines.append(i)
    start_lines.append(len(lines))
    
    # 出力ファイルに保存
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # span行間隔で処理
        for k in range(span):
            m_num = 1
            for i in range(k, len(lines), span):
                # 現在の行
                current_line = lines[i].strip()
                # span行先のトークン列取得
                if(i+span < start_lines[m_num]):
                    tokens = lines[i+span].split("TRACK_START")[0].strip()
                    current_line = current_line + " " + tokens
                else:
                    m_num += 1
                outfile.write(current_line + "\n")

def create_tokenizer(files):

    # Create, train and save the tokenizer.
    print("Preparing tokenizer...")
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(files=files, trainer=trainer)
    return tokenizer

def create_sequence(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
        
    lines = [line.rstrip("\n") for line in lines]
    #楽曲最後の無音部分消去
    del_mode = True
    for i in range(len(lines)-1, -1, -1):
        if del_mode == True and "NOTE_ON" not in lines[i]:
            del lines[i]
        elif "NOTE_ON" in lines[i]:
            del_mode = False
            
        if "PIECE_START" in lines[i]:
            del_mode = True
                
    #楽曲開始位置
    startlist = []
    #PIECE_START消去
    for i in range(0, len(lines), 25):
        tokens = lines[i].split()
        if tokens[0] == "BarCount=0":
            lines[i] = lines[i].replace("PIECE_START ", "")
            if i != 0:
                startlist.append(i)
        
    #PIECE_END追加
    for pos in startlist:
        for i in range(pos-25, pos):
            lines[i] = lines[i] + " PIECE_END"
    
    #オーバーラップ
    """modified_lines = []
    start_pos = 0
    for end_pos in startlist:
        #各楽曲のトークン作成
        for k in range(25):
            pre_line = lines[start_pos+k]
            for line_pos in range(start_pos+k, end_pos, 25):
                if line_pos != start_pos+k:
                    modified_lines.append(pre_line + " " + lines[line_pos] + "\n")
                    pre_line = lines[line_pos]
        start_pos = end_pos"""
        
    #１楽曲を１行に
    """modified_lines = []
    start_pos = 0
    for end_pos in startlist:
        #各楽曲のトークン作成
        for k in range(25):
    """
    
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)
    
def remove_atmosphere(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    modified_lines = []
    for line in lines:
        tokens = line.split()
        # Atmosphere= で始まるトークンを削除
        filtered_tokens = [token for token in tokens if not token.startswith("Atmosphere=")]
        modified_lines.append(" ".join(filtered_tokens) + "\n")
        
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)

def AtmosphereAll_to_AtmosphereAll2(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    Atmosphere_list = []
    modified_lines = []
    for line in lines:
        tokens = line.split()
        #楽曲切り替え部分
        if "BarCount=0" in tokens:
            Atmosphere_list = []
        #Atmosphere追加
        for token in tokens:
            if token == "TRACK_START":
                break
            if token.startswith("Atmosphere"):
               Atmosphere_list.append(token) 
        for i in range(len(Atmosphere_list) - 1, 0, -1):
            if Atmosphere_list[i] == Atmosphere_list[i - 1]:
                del Atmosphere_list[i]
        # AtmosphereListのトークンを最初に追加
        new_tokens = Atmosphere_list + tokens
        modified_lines.append(" ".join(new_tokens) + "\n")
        
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)

def AtmosphereAll2_to_4bar(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    #PIECE_START除去
    for i in range(len(lines)-1):
        lines[i] = lines[i].replace("PIECE_START ", "")
        if "BarCount=0" in lines[i+1]: #PIECE_END追加
            lines[i] = lines[i].rstrip("\n") + " PIECE_END" + "\n"
    lines[len(lines)-1] = lines[len(lines)-1].replace("PIECE_START ", "").rstrip("\n") + " PIECE_END" + "\n"
    
    #4行ずつ抽出(スライディングウィンドウ2行)
    #Atmosphere_list = []
    modified_lines = []
    i = 0
    while i < len(lines)-1:
        if "PIECE_END" in lines[i]: #曲の最後
            i += 1
            continue
        elif "PIECE_END" in lines[i+1]: #曲の最後
            after_bar = "BarCount" + lines[i+1].split("BarCount")[-1].rstrip("\n")
            modified_lines.append(lines[i].rstrip("\n")+" "+after_bar+"\n")
            i += 2
        else: #最後でない時
            after_2bar = "BarCount" + lines[i+2].split("BarCount",1)[1].rstrip("\n")
            modified_lines.append(lines[i].rstrip("\n")+" "+after_2bar+"\n")
            i += 2
              
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)    

"""AtmosphereAll2の小節数を増やす"""
def bar_to_2bar(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    #2小節で1行に
    modified_lines = []
    for i in range(len(lines)-1):
        if "BarCount=0" in lines[i+1]: #楽曲終了
            continue
        elif "BarCount=0" in lines[i]: #楽曲開始
            line = "BarCount"+lines[i].split("BarCount")[1].rstrip("\n")+" BarCount"+lines[i+1].split("BarCount")[1]
        else:
            line = lines[i-1].split("BarCount")[0]+"BarCount"+lines[i].split("BarCount")[1].rstrip("\n")+" BarCount"+lines[i+1].split("BarCount")[1]
        modified_lines.append(line)
        
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)

def revise_train(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    modified_lines = []
    for i in range(len(lines)):
        if "BarCount=0" in lines[i]: #楽曲の最初
            line = "BarCount" + lines[i].split("BarCount",1)[1]
        else:
            line = lines[i-1].split("BarCount")[0]+"BarCount"+lines[i].split("BarCount",1)[1]
        modified_lines.append(line)
    
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(modified_lines)   

"""最初にAtmosphere追加"""
def add_Atmosphere(input_file):
    # ファイルを読み込み、行ごとに処理する
    with open(input_file, "r") as file:
        lines = file.readlines()
    Atmosphere_list = []
    for i in range(len(lines)):
        if "BarCount=0" in lines[i]: #楽曲の最初
            Atmosphere_list = []
        else:
            if "Atmosphere" not in lines[i].split("BarCount")[0]:
                Atmosphere_list.extend(re.findall(r'\bAtmosphere=\d+',lines[i-1].split("BarCount",1)[1]))
                lines[i] = " ".join(Atmosphere_list) + " " + lines[i]
    
    # 結果を新しいファイルに書き込む
    with open(input_file, "w") as file:
        file.writelines(lines)  

#全体クラスタリングの確認
def AtmosCSV(filename):
    df = pd.read_csv(filename)
    # 欠損値を含む行を取得
    rows_with_nan = df[df.isnull().any(axis=1)]
    print(rows_with_nan)
    #df["29"] = df["29"].fillna(df["29"].mean())
    #df.to_csv("Atmosphere_label.csv", index=False)

#クラスタ分析
def Atmosphere_analysis(filename):
    df = pd.read_csv(filename, encoding="UTF-8")
    song_clusters_dict = df.groupby('filename')['cluster'].apply(lambda x: sorted(set(x))).to_dict()
    print(song_clusters_dict)

token_dir = "/mnt/c/Users/yuki-/.vscode/Python/research/MMM/datasets/AtmosphereAll_ver2"
train_path = os.path.join(token_dir, 'token_sequences_train.txt')
valid_path = os.path.join(token_dir, 'token_sequences_valid.txt')

Atmosphere_analysis("AtmosphereAll_label.csv")

#AtmosCSV("Atmosphere_label.csv")
#create_sequence(train_path)
#create_sequence(valid_path)
#remove_atmosphere(train_path)
#remove_atmosphere(valid_path)
#AtmosphereAll_to_AtmosphereAll2(train_path)
#AtmosphereAll_to_AtmosphereAll2(valid_path)

#bar_to_2bar(train_path)
#print("END bar_to_2bar")
#bar_to_2bar(valid_path)
#print("END bar_to_2bar")
#AtmosphereAll2_to_4bar(train_path)
#print("END AtmosphereAll2_to_4bar")
#AtmosphereAll2_to_4bar(valid_path)
#print("END AtmosphereAll2_to_4bar")
#add_Atmosphere(train_path)
#print("END add_Atmosphere")
#add_Atmosphere(valid_path)
#print("END add_Atmosphere")

#tokenizer = create_tokenizer([train_path, valid_path])
#tokenizer_path = os.path.join(token_dir, "tokenizer.json")
#tokenizer.save(tokenizer_path)
#print(f"Saved tokenizer to {tokenizer_path}.")

#process_text_file(train_file,"/mnt/aoni04/obara/MMM-JSB/test.txt")
#process_text_file(valid_file,"/mnt/aoni04/obara/MMM-JSB/token_sequences_valid.txt",1)
