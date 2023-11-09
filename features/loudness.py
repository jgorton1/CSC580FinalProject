# Loudness
# Author: Julius Gorton 
import numpy as np

def rms(audio, start,end):
    np.sqrt(np.mean(audio[start:end] ** 2))

# TODO - there are other algorithms relating to perceived loudness
# also, loudness at given frequencies?
