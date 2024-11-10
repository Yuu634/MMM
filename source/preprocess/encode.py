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

import itertools
import numpy as np
import random
import json
import sys
from source.preprocess.Atmosphere import get_Atmosphere


def encode_songs_data(songs_data, source_path, IsAtmosphere, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill):

    # This will be returned.
    token_sequences = []
    
    if IsAtmosphere == False:
        # Go through all songs.
        for song_data in songs_data:
            token_sequences += encode_song_data(song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill)
    else:
        #Atmosphereトークン取得
        print("Cluster Start")
        Atmosphere_lists = get_Atmosphere(source_path)
        print("Cluster End")
        if Atmosphere_lists == []:
            print("Atmosphereを取得できません")
            sys.exit()
        
        # Go through all songs.
        for song_data in songs_data:
            for Atmosphere_list in Atmosphere_lists:
                for filename,atmosphere in Atmosphere_list.items():
                    if filename == song_data["filename"]:
                        token_sequences += encode_Atmosphere_data(atmosphere, song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill)
                        break
    # Done.
    return token_sequences

#Atmosphere無しのエンコード
def encode_song_data(song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill):

    # This will be returned.
    token_sequences = []
    
    # Start with the tokens.
    token_sequence = ["PIECE_START"]

    # Count the bars.
    bars = get_bars_number(song_data)

    # For iterating over the bars.
    bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)

    # Go through all combinations.
    count = 0
    for (bar_start_index, bar_end_index), transposition in itertools.product(bar_indices, transpositions):
        # Do bar fill if necessary.
        if bar_fill:
            track_data = random.choice(song_data["tracks"])
            bar_data = random.choice(track_data["bars"][bar_start_index:bar_end_index])
            bar_data_fill = {"events": bar_data["events"]}
            bar_data["events"] = "bar_fill"

        # Get the indices. Permute if necessary.
        track_data_indices = list(range(len(song_data["tracks"])))
        if permute:
            random.shuffle(track_data_indices)

        # Encode the tracks.
        for track_data_index in track_data_indices:
            track_data = song_data["tracks"][track_data_index]

            # Encode the track. Insert density tokens. Also transpose.
            encoded_track_data = encode_track_data(track_data, density_bins, bar_start_index, bar_end_index, transposition)
            token_sequence += encoded_track_data

        # Encode the fill tokens.
        if bar_fill:
            token_sequence += encode_bar_data(bar_data_fill, transposition, bar_fill=True)

        #token_sequences += [token_sequence]
        count += 1
        #次の小節に移るトークン追加
        token_sequence += ["NEXTBAR"]

    token_sequence += ["PIECE_END"]
    token_sequences += [token_sequence]
    # Done
    return token_sequences

#Atmosphereありのエンコード
def encode_Atmosphere_data(atmosphere, song_data, transpositions, permute, window_size_bars, hop_length_bars, density_bins, bar_fill):
    token_sequence = []
    
    #曲全体のAtmosphereを挿入
    for time,token in atmosphere.items():
        if token[0] != None:
            token_sequence += ["Atmosphere=" + str(token[0])]
        token_sequence += ["Atmosphere2=" + str(token[1])]
            
    # Start with the tokens.
    token_sequence += ["PIECE_START"]
    # Count the bars.
    bars = get_bars_number(song_data)
    
    bartime = song_data['bartime']

    # For iterating over the bars.
    bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)

    # Go through all combinations.
    count = 0
    maximum = -1
    
    for (bar_start_index, bar_end_index), transposition in itertools.product(bar_indices, transpositions):
        
        """Atmosphere切り替えタイミングと近ければトークン追加"""
        #middletime = ((bar_start_index+bar_end_index)/2)*bartime
        #更新されるAtmosphereの中で最小時間にいるキー取得
        #max_key = min((t for t in atmosphere.keys() if maximum < t <= middletime), default=-1)
        #小節の時間内のキー取得
        for time,atmos_tuple in atmosphere.items():
            if bar_start_index*bartime <= time < bar_end_index*bartime:
                if atmos_tuple[0] != None:
                    token_sequence += ["Atmosphere=" + str(atmos_tuple[0])]
                token_sequence += ["Atmosphere2=" + str(atmos_tuple[1])]

        # Do bar fill if necessary.
        if bar_fill:
            track_data = random.choice(song_data["tracks"])
            bar_data = random.choice(track_data["bars"][bar_start_index:bar_end_index])
            bar_data_fill = {"events": bar_data["events"]}
            bar_data["events"] = "bar_fill"

        # Get the indices. Permute if necessary.
        track_data_indices = list(range(len(song_data["tracks"])))
        if permute:
            random.shuffle(track_data_indices)

        # Encode the tracks.
        for track_data_index in track_data_indices:
            track_data = song_data["tracks"][track_data_index]

            # Encode the track. Insert density tokens. Also transpose.
            encoded_track_data = encode_track_data(track_data, density_bins, bar_start_index, bar_end_index, transposition)
            token_sequence += encoded_track_data

        # Encode the fill tokens.
        if bar_fill:
            token_sequence += encode_bar_data(bar_data_fill, transposition, bar_fill=True)

        #token_sequences += [token_sequence]
        count += 1
        #次の小節に移るトークン追加
        token_sequence += ["NEXTBAR"]

    token_sequence += ["PIECE_END"]
    token_sequences = [token_sequence]
    # Done
    return token_sequences
    

def encode_track_data(track_data, density_bins, bar_start_index, bar_end_index, transposition):

    tokens = []

    tokens += ["TRACK_START"]

    # Set instrument.
    number = track_data["number"]

    # Set the instrument if it is a harmonic one.
    if not track_data.get("drums", False):
        tokens += [f"INST={number}"]

    # Set the instrument if it is drums. Also do not transpose drums.
    else:
        tokens += ["INST=DRUMS"]
        transposition = 0

    # Count note on events.
    note_on_events = 0
    for bar_data in track_data["bars"][bar_start_index:bar_end_index]:
        if bar_data["events"] == "bar_fill":
            continue
        for event_data in bar_data["events"]:
            if event_data["type"] == "NOTE_ON":
                note_on_events += 1

    # Determine density.
    density = np.digitize(note_on_events, density_bins)
    tokens += [f"DENSITY={density}"]

    # Encode the bars.
    for bar_data in track_data["bars"][bar_start_index:bar_end_index]:
        tokens += encode_bar_data(bar_data, transposition)

    tokens += ["TRACK_END"]

    return tokens


def encode_bar_data(bar_data, transposition, bar_fill=False):
    tokens = []

    if not bar_fill:
        tokens += ["BAR_START"]
    else:
        tokens += ["FILL_START"]

    if bar_data["events"] == "bar_fill":
        tokens += ["FILL_IN"]
    else:
        for event_data in bar_data["events"]:
            tokens += [encode_event_data(event_data, transposition)]

    if not bar_fill:
        tokens += ["BAR_END"]
    else:
        tokens += ["FILL_END"]

    return tokens


def encode_event_data(event_data, transposition):
    if event_data["type"] == "NOTE_ON":
        return event_data["type"] + "=" + str(event_data["pitch"] + transposition)
    elif event_data["type"] == "NOTE_OFF":
        return event_data["type"] + "=" + str(event_data["pitch"] + transposition)
    elif event_data["type"] == "TIME_DELTA":
        return event_data["type"] + "=" + str(event_data["delta"])


def get_density_bins(songs_data, window_size_bars, hop_length_bars, bins):

    # Go through all songs and count the note on events for each window.\
    distribution = []
    for song_data in songs_data:
        # Count the bars.
        bars = get_bars_number(song_data)

        # Iterate over over the tracks and the bars.
        bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)
        for track_data in song_data["tracks"]:
            for bar_start_index, bar_end_index in bar_indices:

                # Go through the bars and count notes.
                count = 0
                for bar in track_data["bars"][bar_start_index:bar_end_index]:
                    count += len([event for event in bar["events"] if event["type"] == "NOTE_ON"])

                # Do not count empty tracks.
                if count != 0:
                    distribution += [count]

    # Compute the quantiles, which will become the density bins.
    quantiles = []
    for i in range(100 // bins, 100, 100 // bins):
        if len(distribution) == 0:
            continue
        else:
            quantile = np.percentile(distribution, i)
            quantiles += [quantile]
    return quantiles


def get_density_bins_from_json_files(json_paths, window_size_bars, hop_length_bars, bins):

    # Go through all songs and count the note on events for each window.\
    distribution = []
    for json_path in json_paths:

        # Open the file and get the data.
        song_data = json.load(open(json_path, "r"))

        # Count the bars.
        bars = get_bars_number(song_data)

        # Iterate over over the tracks and the bars.
        bar_indices = get_bar_indices(bars, window_size_bars, hop_length_bars)
        for track_data in song_data["tracks"]:
            for bar_start_index, bar_end_index in bar_indices:

                # Go through the bars and count notes.
                count = 0
                for bar in track_data["bars"][bar_start_index:bar_end_index]:
                    count += len([event for event in bar["events"] if event["type"] == "NOTE_ON"])

                # Do not count empty tracks.
                if count != 0:
                    distribution += [count]

    # Compute the quantiles, which will become the density bins.
    quantiles = []
    for i in range(100 // bins, 100, 100 // bins):
        quantile = np.percentile(distribution, i)
        quantiles += [quantile]
    return quantiles


def get_bars_number(song_data):
    bars = [len(track_data["bars"]) for track_data in song_data["tracks"]]
    bars = max(bars)
    return bars


def get_bar_indices(bars, window_size_bars, hop_length_bars):
    return list(zip(range(0, bars, hop_length_bars), range(window_size_bars, bars, hop_length_bars)))
