import audio_functions
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

duration = 5.0
fs = 44100
buffer_length = 1024

input('Press Enter when ready to record a sample')
samples = audio_functions.record(duration, fs, buffer_length)
print('Finished recording, now plotting recorded audio')#
print(samples)

f, psd = signal.welch(samples, fs, nperseg=1024)

