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

import music21
from music21 import corpus,tempo
from source import logging
from source.preprocess.preprocessutilities import events_to_events_data
from pathlib import Path
import os
import sys
import platform

logger = logging.create_logger("music21jsb")


def preprocess_music21(source_path):

    logger.info("Loading songs...")
    #MIDIデータに変更
    #songs = list(corpus.chorales.Iterator())
    
    ssh_env_vars = ['SSH_CONNECTION', 'SSH_CLIENT', 'SSH_TTY']
    #SSH接続時
    IsSSH = False
    for var in ssh_env_vars:
        if os.getenv(var):
            IsSSH = True
            directory = Path('/mnt/aoni04/obara/MMM-JSB/' + source_path)
            songs = [(f.name, music21.converter.parse(os.path.join('/mnt/aoni04/obara/MMM-JSB',source_path,f.name)) ) for f in directory.iterdir() if f.name.endswith(('.mid','.midi')) ]
            break
            """directory = Path('/mnt/aoni04/obara/MMM-JSB/' + source_path)
            songs = []
            for f in directory.iterdir():
                if f.is_file():
                    try:
                        song = music21.converter.parse(os.path.join('/mnt/aoni04/obara/MMM-JSB',source_path,f.name))
                        songs.append(song)
                    except:
                        print("KeyError")
            break"""
    
    if IsSSH == False:
        if platform.system() == "Windows":
            directory = Path('C:/Users/yuki-/.vscode/Python/research/MMM-JSB/' + source_path)
            songs = [(f.name, music21.converter.parse(os.path.join('C:/Users/yuki-/.vscode/Python/research/MMM-JSB',source_path,f.name)) ) for f in directory.iterdir() if f.name.endswith(('.mid','.midi')) ]
        elif platform.system() == "Linux":
            directory = Path('mnt/c/Users/yuki-/.vscode/Python/research/MMM-JSB/' + source_path)
            songs = [(f.name, music21.converter.parse(os.path.join('mnt/c/Users/yuki-/.vscode/Python/research/MMM-JSB',source_path,f.name)) ) for f in directory.iterdir() if f.name.endswith(('.mid','.midi')) ]
        else:
            sys.exit()  

    logger.info(f"Got {len(songs)} songs.")

    split_index = int(0.8 * len(songs))
    songs_train = songs[:split_index]
    songs_valid = songs[split_index:]
    logger.info(f"Using {len(songs_train)} songs for training.")
    logger.info(f"Using {len(songs_valid)} songs for validation.")

    songs_data_train = preprocess_music21_songs(songs_train, train=True)
    songs_data_valid = preprocess_music21_songs(songs_valid, train=False)

    return songs_data_train, songs_data_valid


def preprocess_music21_songs(songs, train):
    #print("SONGS")

    songs_data = []
    for filename,song in songs:
        song_data = preprocess_music21_song(song, train, filename)
        if song_data is not None:
            songs_data += [song_data]

    return songs_data

#メタデータ取得
def preprocess_music21_song(song, train, filename):
    #print("  SONG", song.metadata.title, song.metadata.number)

    # Skip everything that has multiple measures and/or are not 4/4.
    meters = [meter.ratioString for meter in song.recurse().getElementsByClass(music21.meter.TimeSignature)]
    meters = list(set(meters))
    bpm = None
    for part in song.parts:
        for element in part.recurse():
            if isinstance(element, music21.tempo.MetronomeMark):
                bpm = element.number
                beats_per_bar = int(meters[0].split('/')[0])
                bartime = (60 / bpm) * beats_per_bar
    #measureCount = song.duration.quarterLength / 4
    
    if len(meters) != 1:
        logger.debug(f"Skipping because of multiple measures.")
        return None
    elif meters[0] != "4/4":
        logger.debug(f"Skipping because of meter {meters[0]}.")
        return None
    elif bpm == None:
        logger.debug(f"Don't Exist BPM.")
        return None
    """elif measureCount < 32:
        logger.debug("Skipping because of short measure.")
        return None
    """

    song_data = {}
    song_data["filename"] = filename.replace('.mid','.wav').replace('.midi','.wav')
    song_data["bartime"] = bartime
    #song_data["title"] = song.metadata.title
    #song_data["number"] = song.metadata.number
    song_data["tracks"] = []
    for part_index, part in enumerate(song.parts):
        track_data = preprocess_music21_part(part, part_index, train)
        song_data["tracks"] += [track_data]

    return song_data

#トラック情報付与
def preprocess_music21_part(part, part_index, train):
    #print("    PART", part.partName)

    track_data = {}
    track_data["name"] = part.partName
    track_data["number"] = part_index
    track_data["bars"] = []
    #print(track_data)

    for measure_index in range(1,1000):
        measure = part.measure(measure_index)
        if measure is None:
            break
        bar_data = preprocess_music21_measure(measure, train)
        track_data["bars"] += [bar_data]
    return track_data

#音楽トークン付与
def preprocess_music21_measure(measure, train):
    #print("      MEASURE")

    bar_data = {}
    bar_data["events"] = []

    events = []
    for note in measure.recurse(classFilter=("Note")):
        #print("        NOTE", note.pitch.midi, note.offset, note.duration.quarterLength)
        events += [("NOTE_ON", note.pitch.midi, 4 * note.offset)]
        events += [("NOTE_OFF", note.pitch.midi, 4 * note.offset + 4 * note.duration.quarterLength)]

    bar_data["events"] = events_to_events_data(events)
    return bar_data

    events = sorted(events, key=lambda event: event[2])
    for event_index, event, event_next in zip(range(len(events)), events, events[1:] + [None]):
        if event_index == 0 and event[2] != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": event[2]
            }
            bar_data["events"] += [event_data]

        event_data = {
            "type": event[0],
            "pitch": event[1]
        }
        bar_data["events"] += [event_data]

        if event_next is None:
            continue

        delta = event_next[2] - event[2]
        assert delta >= 0, events
        if delta != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": delta
            }
            bar_data["events"] += [event_data]

    return bar_data
