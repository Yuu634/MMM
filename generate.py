import tensorflow as tf
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os
import numpy as np
from source.helpers.samplinghelpers import *
from itertools import takewhile
import sys

#datapath = "ver1"
# Where the checkpoint lives.
# Note can be downloaded from: https://ai-guru.s3.eu-central-1.amazonaws.com/mmm-jsb/mmm_jsb_checkpoints.zip
#check_point_path = os.path.join("checkpoints", "20210411-1426")

#print("Reconstructed sample")
#render_token_sequence(generated_sample, use_program=False)


#楽曲全体の作成
def generate_song(data_path,Atmosmodel_name,Barmodel_name):
    """Atmosphere生成準備"""
    validation_data_path = os.path.join("datasets", data_path, Atmosmodel_name, "token_sequences_valid.txt")

    #Load tokenizer
    tokenizer_path = os.path.join("datasets", data_path, Atmosmodel_name, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    #Load model
    #model_path = os.path.join("training", data_path, Atmosmodel_name, "checkpoint-181")
    model_path = os.path.join("training", data_path, "checkpoint-163000")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    """Atmosphere生成"""
    print("Model loaded.")
    priming_sample, priming_sample_original = get_priming_token_sequence(
        validation_data_path,
        stop_on_track_end=0,
        stop_after_n_tokens=20,
        return_original=True
    )
    #priming_sample = "Atmosphere=3 Atmosphere2=3 Atmosphere2=3 Atmosphere2=1"
    generated_sample = generate(model, tokenizer, priming_sample)
    print(priming_sample)
    print(generated_sample)
    print("Atmosphere generated.")
    sys.exit()
    
    #Atmosphereの分割数決定
    
    """Bar生成準備"""
    validation_data_path = os.path.join("datasets", data_path, Barmodel_name, "token_sequences_valid.txt")

    #Load tokenizer
    tokenizer_path = os.path.join("datasets", data_path, Barmodel_name, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    #Load model
    #model_path = os.path.join("training", data_path, Barmodel_name, "best_model")
    model_path = os.path.join("training", data_path, Barmodel_name, "best_model")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    """Bar生成"""
    generated_sample = list(takewhile(lambda x: x != '[PAD]', generated_sample.split()))
    song_sequences = ""
    bar_num = round(len(generated_sample)/4)
    for i in range(0,len(generated_sample),4):
        token_list = generated_sample[i:i+4]
        condition = ' '.join(token_list)
        bar_sequences = generate(model, tokenizer, condition)
        stop_index = bar_sequences.find('[PAD]')
        if stop_index == -1:
            song_sequences += bar_sequences
        else:
            song_sequences += bar_sequences[:stop_index]
        print(song_sequences)
        print("Bar generated : " + str(round(i/4+1)) + '/' + str(bar_num))
    
    print(song_sequences)
    return song_sequences


generate_song("AtmosphereAll", "BarModel", "BarModel")