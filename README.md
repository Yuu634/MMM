# MMM
Atmosphere2:Atmosphereのクラスタ数の２倍でクラスタリングした結果 <br>
Dataset:vgDataset全体 <br>
データ制限:4/4拍子、拍子変化なし <br>

config: <br>
dataset_name="ver1", #保存先ディレクトリ <br>
datasetSource_path='vgDataset', #学習データセットディレクトリ <br>
IsAtmosphere=True, #Atmosphereトークンを付けるか <br>
encoding_method="mmmtrack", <br>
#json_data_method="My_preprocess", <br>
json_data_method="preprocess_music21", <br>
window_size_bars=2, #何小節で分割するか <br>
hop_length_bars=2, #何小節スライドするか <br>
density_bins_number=5, <br>
transpositions_train=list(range(-12, 13)), <br>
permute_tracks=False, #トークン生成をトラック順に行うか（Falseで順番に） <br>
**kwargs <br>
