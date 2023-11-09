# Zero-crossing rate
# Author: Julius Gorton


def zcr(audio, start,end, sample_rate =22050):
    # assume 22050 hz 
    zerocrossings = 0
    for i in range(start,end-1):
        if (audio[i] > 0 and audio[i+1] <=0):
            zerocrossings += 1
    return zerocrossings / ((end-start) /sample_rate)