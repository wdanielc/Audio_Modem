import numpy as np
import audio_functions as audio
import encoding_functions as encode
from config import *

symbol_length = 1024

QAM_values = np.ones(symbol_length, dtype=np.complex)

transmit = np.tile(encode.OFDM(QAM_values, Lp, Fc, Fs, dF), 2)
transmit = np.insert(transmit,0,encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF))

audio.play(transmit, volume, Fs)
