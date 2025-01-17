import tensorflow as tf
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os
import numpy as np
from source.helpers.samplinghelpers import *
import sys

# Where the checkpoint lives.
# Note can be downloaded from: https://ai-guru.s3.eu-central-1.amazonaws.com/mmm-jsb/mmm_jsb_checkpoints.zip
#check_point_path = os.path.join("checkpoints", "20210411-1426")

def create_music(model_path):
    # Load the validation data.
    validation_data_path = os.path.join("datasets", model_path, "token_sequences_valid.txt")

    # Load the tokenizer.
    tokenizer_path = os.path.join("datasets", model_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load the model.
    model_path = os.path.join("training", model_path, "best_model")
    model = GPT2LMHeadModel.from_pretrained(model_path)

    print("Model loaded.")
    priming_sample, priming_sample_original = get_priming_token_sequence(
        validation_data_path,
        stop_on_track_end=0,
        stop_after_n_tokens=20,
        return_original=True
    )
    priming_sample = "PIECE_START TRACK_START INST=2 DENSITY=1 BAR_START NOTE_ON=64 TIME_DELTA=12.0 NOTE_OFF=64 NOTE_ON=64 TIME_DELTA=2.0 NOTE_OFF=64 NOTE_ON=62 TIME_DELTA=2.0 NOTE_OFF=62 BAR_END BAR_START NOTE_ON=60 TIME_DELTA=4.0 NOTE_OFF=60 NOTE_ON=63 TIME_DELTA=4.0 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=4.0 NOTE_OFF=62 NOTE_ON=67 TIME_DELTA=4.0 NOTE_OFF=67 BAR_END TRACK_END TRACK_START INST=3 DENSITY=3 BAR_START NOTE_ON=52 TIME_DELTA=12.0 NOTE_OFF=52 NOTE_ON=60 TIME_DELTA=2.0 NOTE_OFF=60 NOTE_ON=59 TIME_DELTA=2.0 NOTE_OFF=59 BAR_END BAR_START NOTE_ON=57 TIME_DELTA=2.0 NOTE_OFF=57 NOTE_ON=55 TIME_DELTA=2.0 NOTE_OFF=55 NOTE_ON=53 TIME_DELTA=4.0 NOTE_OFF=53 NOTE_ON=58 TIME_DELTA=2.0 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=2.0 NOTE_OFF=57 NOTE_ON=55 TIME_DELTA=4.0 NOTE_OFF=55 BAR_END TRACK_END"
    generated_sample = generate(model, tokenizer, priming_sample)
    print(priming_sample)
    print(generated_sample)

    #print("Original sample")
    #ender_token_sequence(priming_sample_original, use_program=False)

    #print("Reduced sample")
    #render_token_sequence(priming_sample, use_program=False)

    print("Reconstructed sample")
    render_token_sequence(generated_sample, use_program=False)

create_music("jsb_mmmtrack")