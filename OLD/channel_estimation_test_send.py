import audio_functions
import numpy as np

volume = 1.0
fs = 44100
duration = 5.0

sigma = 0.2 # standard deviation of Gaussian (square root of variance)
Nsamples = int(duration*fs)

gaussian = (np.clip(np.random.normal(0.0,sigma,Nsamples),-1.0, 1.0)).astype(np.float32)		#generate gaussian white noise
audio_functions.play(gaussian, volume, fs)

