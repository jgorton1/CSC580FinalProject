import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from os import listdir
from os.path import isfile
from typing import Any
import multiprocessing as mp
from multiprocessing import SimpleQueue
import json as js








def getChroma(filename: str):
    #print(len(frequencies))
    #print(times)
    #print(len(spectrogram[0]))
    #plt.pcolormesh(times, frequencies, np.log(spectrogram), cmap="plasma")
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    '''
    Simple conversion code for going from spectrogram of frequencies 
    '''
    sample_rate, samples = wavfile.read('Data\\train\\classical\\classical.00072.wav')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=4400, noverlap=0)

    octaveCount: int = 1
    stepCount: int = 0
    c1_freq = 32.7032 # can get the exact later, derive from A4 = 440
    threshold = .5

    cutoff_array: list[float] = []
    for x in range(1, 10):
        for y in range(0, 12):
            lower = c1_freq * 2 ** ((x - 1) + (y - threshold) / 12 )
            cutoff_array.append(lower)
            if(x == 9 and y == 11):
                upper = c1_freq * 2 ** ((x - 1) + (y + threshold) / 12 )
                cutoff_array.append(upper)


    # this is a merge-like algorithm, frequencies and cutoffs must be sorted
    map: dict[int, list[int]] = {}
    index = 0
    for x in range(len(frequencies)):
        curFreq = frequencies[x]
        if(curFreq < cutoff_array[0]):
            # subsonic
            # print("Unhearable (basically)")
            if(-1 not in map):
                map[-1] = [x]
            else:
                map[-1].append(x)
            continue

        while(curFreq > cutoff_array[index]):
            index += 1
        
        if(index not in map):
            map[index] = [x]
        else:
            map[index].append(x)

    # each range of x requires the spectrogram to be combined, question is, multiplicative or additive
    maxNote = max(map.keys())
    new_amps: list[list[float]] = []
    for x in range(0, maxNote + 1):
        temp_amps: list[float] = []
        if(x not in map):
            for counter in range(len(times)):
                temp_amps.append(0.0001)
            new_amps.append(temp_amps)
            continue


        y = map[x]
        for counter in range(len(times)):
            temp_amp: float = 0
            for freq in y:
                temp_amp += spectrogram[freq][counter]
            temp_amps.append(temp_amp)
        new_amps.append(temp_amps)


    final_amps: list[list[float]] = [[],[],[],[],[],[],[],[], [], [], [], []]
    for step in range(len(times)):
        counter = 0
        simplified_amps: list[float] = [0,0,0,0,0,0,0,0,0,0,0,0]
        for note in new_amps:
            key = counter % 12
            simplified_amps[key] += note[step] # if a note is not being played, will multiply cuz log...
            counter += 1
        for x in range(len(final_amps)):
            final_amps[x] += [simplified_amps[x]]
        
    return (final_amps)


'''
# harmonic features
for step in range(len(times)):
    temp_list = []
    for x in range(len(final_amps)):
        temp_list.append((x, final_amps[x][step]))
    get_harmonic_vals(temp_list)
'''


def __main__():
    test = ".\\Data\\test\\"
    train = ".\\Data\\train\\"
    trainDirectories = [f for f in listdir(train)]
    testDirectories = [f for f in listdir(test)]
    print(testDirectories)
    
    data = {}
    data["train"] = {}
    for name in trainDirectories:
        files = [f for f in listdir(train + name)]
        data["train"][name] = {}
        for filename in files:
            num: int = int(filename.split(".")[1])
            
            temp = getChroma(train + name + filename)
            data["train"][name][num] = temp
    
    data["test"] = {}
    for name in testDirectories:
        files = [f for f in listdir(test + name)]
        data["test"][name] = {}
        for filename in files:
            num: int = int(filename.split(".")[1])
            
            temp = getChroma(test + name + filename)
            data["test"][name][num] = temp

    cgfile = open(".\\Data\\chronogram.json", "w")
    js.dump(data, cgfile)

if __name__ == '__main__':
    __main__()
    







    

    
    
    



