#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:15:46 2019

@author: wdc24
"""
import encoding_functions as encode
import audio_functions as audio
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
from scipy import signal

symbol_length = 32
volume = 1.0
fs = 44100

Fc = 10000 # Carrier frequency
dF = 10
T = 1/dF
QAM = 2 # This doesn't fully work yet, leave as 2

# Get random string to test the modulation
letters = "".join(chr(randrange(127)) for i in range(1000))
data = bytes(letters, 'ascii')
transmit = []

# Slice the text into chunks of the correct length (for symbols)
data = [data[i:i+int(symbol_length/((4**QAM)/8))] for i in range(0,len(data),int(symbol_length/((4**QAM)/8)))]

# Encode the data with QAM
for block in data:
    symbol = np.zeros(symbol_length, dtype=np.complex)
    for i in range(0,len(block)*2,2):
        symbol[i] = encode.QAM(block[int(i/2)] >> 4, 2)
        symbol[i+1] = encode.QAM(block[int(i/2)] % 16, 2)
    # This makes sure on playback each symbol takes T seconds
    symbol = encode.interpolate(symbol, fs, T)
    # OFDM the symbol and prepare to transmit
    for value in encode.OFDM(symbol, 2):
        transmit.append(value)

Ts = 1/fs
transmit = encode.upconvert(np.array(transmit), Fc, Ts)

transmit = np.array(transmit,dtype=np.float32) # Convert to correct data type for playback

#plt.figure()
#plt.plot(np.arange(len(transmit)), transmit)

plt.figure()
f, psd = signal.welch(transmit, fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))

audio.play(transmit, volume, fs)

plt.show()