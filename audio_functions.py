import pyaudio as pa
import wave
import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt

def play(samples, volume, fs):
    p = pa.PyAudio()
    stream = p.open(format=pa.paFloat32, channels=1, rate=fs, output=True)
    stream.write(volume*np.tile(samples,4))
    stream.stop_stream()
    stream.close()
    p.terminate()