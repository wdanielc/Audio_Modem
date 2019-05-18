#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:15:46 2019

@author: wdc24
"""
import encoding_functions as encode
import audio_functions as audio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import data_input as data

filename = "hamlet.txt"

symbol_length = 10192
volume = 1.0
fs = 44100

Fc = 10000 # Carrier frequency
dF = 1
T = 1/dF
QAM = 2

# This is a list of QAM values of the data
QAM_values = data.modulate(data.get_data(filename), QAM)

print(len(QAM_values))

transmit = np.array([])

for i in range(int(np.ceil(len(QAM_values)/symbol_length))):
    block = QAM_values[i * symbol_length:(i + 1) * symbol_length]

    ### The block of QAM values needs to be turned into a series of OFDM symbols
    transmit = np.append(transmit, encode.OFDM(block, 300, Fc, fs))

print(len(transmit))

plt.figure()
f, psd = signal.welch(transmit, fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))


audio.play(transmit, volume, fs)

plt.show()
