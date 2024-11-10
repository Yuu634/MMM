# MMM
Atmosphere2:Atmosphereのクラスタ数の２倍でクラスタリングした結果
Dataset:vgDataset全体
データ制限:4/4拍子、拍子変化なし
config:     dataset_name="ver1", #保存先ディレクトリ
            datasetSource_path='vgDataset', #学習データセットディレクトリ
            IsAtmosphere=True, #Atmosphereトークンを付けるか
            encoding_method="mmmtrack",
            #json_data_method="My_preprocess",
            json_data_method="preprocess_music21",
            window_size_bars=2, #何小節で分割するか
            hop_length_bars=2, #何小節スライドするか
            density_bins_number=5,
            transpositions_train=list(range(-12, 13)),
            permute_tracks=False, #トークン生成をトラック順に行うか（Falseで順番に）
            **kwargs
