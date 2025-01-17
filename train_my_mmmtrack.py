# Copyright 2021 Tristan Behrens.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
 
import os
import sys
import torch
from source import datasetcreatorconfig
from source import datasetcreator
from source import mmmtrainerconfig
from source import mmmtrainer

#GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# Myデータセット用訓練コード
# <変更点>
# dataset_name⇒train用データフォルダ名
# json_data_method⇒MIDI表記法pyファイル
# 

dataset_creator_config = datasetcreatorconfig.MyDatasetCreatorTrackConfig()

dataset_creator = datasetcreator.DatasetCreator(dataset_creator_config)

#トークン列が存在しなかったらdataset作成
pathname = "datasets"
MMMdirectory = "/mnt/aoni04/obara/MMM-JSB"
directory_path = os.path.join(MMMdirectory, pathname, dataset_creator_config.dataset_name)
if not os.path.isdir(directory_path):
    print("dataset Create")
    dataset_creator.create(datasets_path=os.path.join(MMMdirectory, pathname), overwrite=False)
# Train the model.
#楽曲全体学習
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", dataset_creator_config.dataset_name, "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", dataset_creator_config.dataset_name, "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", dataset_creator_config.dataset_name, "token_sequences_valid.txt")],
    pad_length=1024,
    n_positions=1024,
    shuffle_buffer_size=10000,
    batch_size=4,
    epochs=5,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
trainer.train(
    output_path=os.path.join("training", dataset_creator_config.dataset_name),
    simulate="simulate" in sys.argv
    )
sys.exit()



"""Atmosphereモデル学習
Atmos_path = "AtmosphereModel"
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "token_sequences_valid.txt")],
    pad_length=128,
    #n_positions=20000,
    shuffle_buffer_size=10000,
    batch_size=2,
    epochs=10,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
print("TrainStart:AtmosphereModel")
trainer.train(
    output_path=os.path.join("training", dataset_creator_config.dataset_name, Atmos_path),
    simulate="simulate" in sys.argv
    )
sys.exit()
"""

"""Barモデル学習"""
Bar_path = "BarModel"
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "token_sequences_valid.txt")],
    pad_length=8192,
    n_positions=8192,
    shuffle_buffer_size=10000,
    batch_size=1,
    epochs=5,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
print("TrainStart:BarModel\n")
trainer.train(
    output_path=os.path.join("training", dataset_creator_config.dataset_name, Bar_path),
    simulate="simulate" in sys.argv
    )
