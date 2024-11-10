import tensorflow as tf
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import os
import numpy as np
from source.helpers.samplinghelpers import *
import pretty_midi

# Where the checkpoint lives.
# Note can be downloaded from: https://ai-guru.s3.eu-central-1.amazonaws.com/mmm-jsb/mmm_jsb_checkpoints.zip
#check_point_path = os.path.join("checkpoints", "20210411-1426")

# Load the validation data.
#validation_data_path = os.path.join(check_point_path, "datasets", "jsb_mmmtrack", "token_sequences_valid.txt")
validation_data_path = os.path.join("datasets", "jsb_mmmtrack", "token_sequences_valid.txt")

# Load the tokenizer.
#tokenizer_path = os.path.join(check_point_path, "datasets", "jsb_mmmtrack", "tokenizer.json")
tokenizer_path = os.path.join("datasets", "jsb_mmmtrack", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the model.
#model_path = os.path.join(check_point_path, "training", "jsb_mmmtrack", "best_model")
model_path = os.path.join("training", "jsb_mmmtrack", "best_model")
model = GPT2LMHeadModel.from_pretrained(model_path)

print("Model loaded.")

priming_sample, priming_sample_original = get_priming_token_sequence(
    validation_data_path,
    stop_on_track_end=0,
    stop_after_n_tokens=20,
    return_original=True
)
print(priming_sample)

generated_sample = generate(model, tokenizer, priming_sample)
print(generated_sample)

#print("Original sample")
#render_token_sequence(priming_sample_original, use_program=False)

#print("Reduced sample")
#render_token_sequence(priming_sample, use_program=False)

#print("Reconstructed sample")
#render_token_sequence(generated_sample, use_program=False)

def Create_midi(tokens, output_file):
    # MIDIオブジェクトの作成
    midi = pretty_midi.PrettyMIDI()
    # インストゥルメントの作成 (ここではピアノ)
    instrument = pretty_midi.Instrument(program=pretty_midi.program_to_program_number(program=))

    current_time = 0  # 開始時間
    track_active = False  # トラックがアクティブかどうか
    track_notes = []  # 現在のトラックのノートリスト
    track_start_time = 0  # トラックの開始時間
    time_delta = 0  # 拍の経過時間
     
    # トークンの処理
    for token in tokens:
        if token == 'PIECE_START':
            pass
        
        # トラック開始
        elif token == 'TRACK_START':
            track_active = True  
            track_notes = []  # トラック内のノートリストをリセット
            track_start_time = current_time  # トラックの開始時間
        
        # トラック終了
        elif token == 'TRACK_END':
            track_active = False 
            if track_notes:
                # トラックに残っているノートをMIDIに追加
                for note in track_notes:
                    instrument.notes.append(note)
            track_notes = []  # リセット

        #音符開始
        elif token.startswith('NOTE_ON='):
            pitch = int(token.split('=')[1])  # ピッチの取得
            start_time = current_time  # 音符の開始時間
            end_time = start_time + 0.5  # 音符の終了時間（デフォルトで0.5拍）
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
            track_notes.append(note)  # トラック内のノートとして追加
        
        #音符終了
        elif token.startswith('NOTE_OFF='):
            pitch = int(token.split('=')[1])  # ピッチの取得
            # 対応するNOTE_ONの終了時間を設定（NOTE_ONの時間に基づいて終わる）
            for note in track_notes:
                if note.pitch == pitch and note.end == start_time:
                    note.end = current_time
                    break

        elif token.startswith('TIME_DELTA='):
            # TIME_DELTA=経過時間（拍数）
            time_delta = float(token.split('=')[1])  # 経過時間
            current_time += time_delta  # 経過時間を加算
            
        elif token == 'BAR_START':
            # 小節の開始は特に処理しない
            pass
        
        elif token == 'BAR_END':
            # 小節の終了も特に処理しない
            pass

    # インストゥルメントをMIDIファイルに追加
    midi.instruments.append(instrument)



    # MIDIファイルとして保存
    midi.write(output_file)
    print(f'MIDIファイルが {output_file} として保存されました！')