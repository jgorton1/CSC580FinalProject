'''
This file takes the chromagraph data as input and creates our custom harmonic and melodic features
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from io import TextIOWrapper
import time
import json as js


#verify data
'''
for x in cgdata:
    print(x)
    for y in cgdata[x]:
        print(y)
        for z in cgdata[x][y]:
            print(z)
            data = cgdata[x][y][z]
            if(len(data) != 12 or len(data[0]) != 150):
                print(len(data), len(data[0]))
        print()
'''






#plt.pcolormesh(range(0, 150), range(0, 12), np.log(bluestwos), cmap='plasma')
#plt.ylabel('Note Number')
#plt.xlabel('Time interval')
#plt.show()

    

def main():
    
    train_file = open(".\\Data\\train_features.csv", "w");
    test_file = open(".\\Data\\test_features.csv", "w");
    write_labels(train_file)
    write_labels(test_file)
    
    
    cgfile = open(".\\Data\\chronogram.json")
    

    cgdata = js.load(cgfile)
    #print(cgdata['train']['blues']['0'][0])
    #print(cgdata['train']['classical']['0'][0])
    
    for x in cgdata:
        for y in cgdata[x]:
            for z in cgdata[x][y]:
                # X is train/test, y is genre, z is number
                data: list[list[float]] = cgdata[x][y][z]
                song_features = get_features_from_song(data)
                # there are 10 partitions in song features
                counter = 0
                for partition in song_features:
                    if(len(song_features) != 10):
                        print(x, y, z)
                    
                    cleaned_up = clean_up(partition)
                    if(x == "train"):
                        write_data(cleaned_up, train_file, y, z, counter)
                    else:
                        write_data(cleaned_up, test_file, y, z, counter)
                    counter += 1
                    
    

def write_labels(file: TextIOWrapper):
 # labels
    harmonies: dict[tuple[int, int], float] = {}
    melodies: dict[int, float] = {}
    for x in range(1, 7):
        for y in range(1, 7 - x // 6):
            harmonies[(x, y)] = 0
    for x in range(0, 7):
        melodies[x] = 0

    file.write(f"Title")
    for x in sorted(harmonies.keys()):
        file.write("," + str(x))
    for y in sorted(melodies.keys()):
        file.write("," + str(y))
    file.write(f",label\n")

def write_data(data: tuple[dict[tuple[int, int], float], dict[int, float]] ,file: TextIOWrapper, name: str, num: str, part: int):
    harmonies: dict[tuple[int, int], float] = data[0]
    melodies: dict[int, float] = data[1]
    file.write(f"{name}.{num}.{part}")
    for x in sorted(harmonies.keys()):
        file.write("," + str(harmonies[x]))
    for y in sorted(melodies.keys()):
        file.write("," + str(melodies[y]))
    file.write(f",{name}\n")


'''
Methods used for extracting features and writing into data files
'''
def get_features_from_song(song_vals: list[list[float]]):
    song: list[list[tuple[list[int], dict[int, float] | None]]] = []
    for x in range(10):
        cur_partition: list[tuple[list[int], dict[int, float] | None]] = []
        prevNotes: None | list[tuple[int, float]] = None
        for y in range(15):
            notes: list[tuple[int, float]] = []
            timeStep: int = (x + 1) * y
            for note in range(len(song_vals)):
                notes.append((note, song_vals[note][timeStep]))
            raw_features = get_features(prevNotes, notes)
            cur_partition.append(raw_features)
            prevNotes = notes
        song.append(cur_partition)
    return song
        
            
def get_features(prev: list[tuple[int, float]] | None, cur: list[tuple[int, float]]) -> tuple[list[int], dict[int, float] | None]:
    harmonic_vals = get_harmonic_vals(cur)
    melodic_vals: dict[int, float] | None = None
    if(prev is not None):
        melodic_vals = get_melodic_vals(prev, cur)
    
    return harmonic_vals, melodic_vals

def get_harmonic_vals(arr: list[tuple[int, float]]):
    # let us only worry about chords of 3 (this will generate a ton of )
    # further, let us only worry about spacing 1 -> 3 -> 5 is the same as 2 -> 4 -> 6
    arr.sort(reverse=True, key=lambda x : x[1])
    chord = arr[:3]
    # organize chord to be in note order
    chord.sort(key=lambda x : x[0])
    first = abs(chord[0][0] - chord[1][0])
    second = abs(chord[1][0] - chord[2][0])
    first = min(first, 12 - first)
    second = min(second, 12 - second)
    return [first, second]

'''
Melodic intervals only make sense when given two steps
'''
def get_melodic_vals(prev: list[tuple[int, float]], cur: list[tuple[int, float]]):
    # same thing, looking at 3 loudest? (what if there is only one voice) Maybe do not include things less than 5 log 
    # this is also multiplicative?
    # could weight it based on amplitude
    prev.sort(reverse=True, key=lambda x : x[1])
    cur.sort(reverse=True, key=lambda x : x[1])
    res: dict[int, float] = {}

    ampTotal: float = 0
    for x in prev[:3]:
        ampTotal += (3* x[1])
    for y in cur[:3]:
        ampTotal += (3 * y[1])

    for x in prev[:3]:
        for y in cur[:3]:
            intensityPrev = x[1]
            pitchPrev = x[0]
            intensityCur = y[1]
            pitchCur = y[0]
            step = abs(pitchCur - pitchPrev)
            step = min(step, 12 - step)
            score = (intensityPrev + intensityCur) / ampTotal
            if(step in res):
                res[step] += score
            else:
                res[step] = score
    return res

def clean_up(partition: list[tuple[list[int], dict[int, float] | None]]):
    harmonies = [x for (x, _) in partition]
    melodies = [y for (_, y) in partition]
    size = len(harmonies)

    # set up all possible features, 55 combinations (similar for melodic, but just 12)
    harmony_dict: dict[tuple[int, int], float] = {}
    melody_dict: dict[int, float] = {}
    for x in range(1, 7):
        for y in range(1, 7 - (x // 6)):
            harmony_dict[(x, y)] = 0
    for x in range(0, 7):
        melody_dict[x] = 0
    
    for x in harmonies:
        harmony_dict[(x[0], x[1])] += 1/size
    for x in melodies:
        if(x is None):
            continue
        # divide by 14 since there are 14 timestamps per partition
        for melody, score in x.items():
            melody_dict[melody] += score / 14
    
    return harmony_dict, melody_dict


    
    


main()