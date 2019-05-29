import numpy as np
import audio_functions as audio
import encoding_functions as encode
from config import *

symbol_length = 1024

QAM_values = np.ones(symbol_length, dtype=np.complex)

white_noise = (np.clip(np.random.normal(0.0,0.2,frame_length + Lp),-1.0, 1.0)).astype(np.float32)

transmit = np.tile(white_noise, 2)
transmit = np.insert(transmit,0,encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF))

audio.play(transmit, volume, Fs)
