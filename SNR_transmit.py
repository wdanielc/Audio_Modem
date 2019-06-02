import numpy as np
import audio_functions as audio
from config import *

white_noise = (np.clip(np.random.normal(0.0,0.2,44100*7),-1.0, 1.0)).astype(np.float32)

input("Press enter to play white noise")

audio.play(white_noise, volume, Fs)

print('End.')

