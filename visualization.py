import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read('Data\\train\\classical\\classical.00008.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=1100, noverlap=0)

#print(len(frequencies))
#print(times)
#print(len(spectrogram[0]))
#plt.pcolormesh(times, frequencies, np.log(spectrogram), cmap="inferno_r")
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

'''
Simple conversion code for going from spectrogram of frequencies 
'''
octaveCount = 1
stepCount = 0
c1_freq = 32.7032 # can get the exact later, derive from A4 = 440
threshold = .5

cutoff_array: list[float] = []
for x in range(1, 10):
    for y in range(0, 12):
        lower = c1_freq * 2 ** ((x - 1) + (y - threshold) / 12 )
        exact = (c1_freq * 2 ** ((x - 1) + (y / 12)))
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
        print("Unhearable (basically)")
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


final_amps = [[],[],[],[],[],[],[],[]]
for step in range(len(times)):
    counter = 0
    simplified_amps: list[float] = [0,0,0,0,0,0,0,0]
    for note in new_amps:
        key = counter % 8
        simplified_amps[key] += note[step] # if a note is not being played, will multiply cuz log...
        counter += 1
    for x in range(len(final_amps)):
        final_amps[x] += [simplified_amps[x]]

#for x in new_amps:
#    print(np.log(x[5]))

plt.pcolormesh(times, range(0, 8), np.log(final_amps))
plt.ylabel('Note Number')
plt.xlabel('Time [sec]')
plt.show()


    

    
    
    



