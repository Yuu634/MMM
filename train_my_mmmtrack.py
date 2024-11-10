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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
local_directory = "C:/Users/yuki-/.vscode/Python/research/MMM-JSB"
directory_path = os.path.join(local_directory, pathname, dataset_creator_config.dataset_name)
if not os.path.isdir(directory_path):
    print("datasetCreate")
    dataset_creator.create(datasets_path=os.path.join(local_directory, pathname), overwrite=False)

# Train the model.
"""Atmosphereモデル学習"""
Atmos_path = "AtmosphereModel"
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Atmos_path, "token_sequences_valid.txt")],
    pad_length=128,
    #n_positions=20000,
    shuffle_buffer_size=10000,
    batch_size=32,
    epochs=10,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
print("TrainStart:AtmosphereModel")
trainer.train(
    output_path=os.path.join("training", dataset_creator_config.dataset_name, Atmos_path),
    simulate="simulate" in sys.argv
    )

"""Barモデル学習"""
Bar_path = "BarModel"
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", dataset_creator_config.dataset_name, Bar_path, "token_sequences_valid.txt")],
    pad_length=256,
    #n_positions=20000,
    shuffle_buffer_size=10000,
    batch_size=16,
    epochs=10,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
print("TrainStart:BarModel")
trainer.train(
    output_path=os.path.join("training", dataset_creator_config.dataset_name, Bar_path),
    simulate="simulate" in sys.argv
    )


# Train the model.
"""trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", "pop_mmmtrack", "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", "pop_mmmtrack", "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", "pop_mmmtrack", "token_sequences_valid.txt")],
    pad_length=20000,
    n_positions=20000,
    shuffle_buffer_size=10000,
    batch_size=1,
    epochs=10,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
trainer.train(
    output_path=os.path.join("training/mytrack"),
    simulate="simulate" in sys.argv
    )
"""
