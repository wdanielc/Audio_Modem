#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:16:19 2019

@author: wdc24
"""

import encoding_functions as encode
import decoding_functions as decode
import audio_functions as audio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import data_input as data
import wave
import channel


filename = "hamlet_output.txt"	#file to save to
Modulation_type_OFDM = True  #True for OFDM, False for DMT


fs = 44100

Fc = 10000 # Carrier frequency
dF = 10
T = 1/dF
QAM = 2
OFDM_symbol_length = 1024
DMT_symbol_length = int(((fs/dF)-2)/2)
Lp = 350


receive = channel.get_received(sigma=0.005, h=True, ISI=False)


if Modulation_type_OFDM:
	symbol_length = OFDM_symbol_length
else:
	symbol_length = DMT_symbol_length

#transmitted_QAM = data.modulate(data.get_data('hamlet_semiabridged.txt'), QAM, OFDM_symbol_length*2*QAM)

data_bits = data.get_data('hamlet_semiabridged.txt') 
frame_length_bits = symbol_length*2*QAM
transmit_frames = int(np.ceil(len(data_bits)/frame_length_bits))

QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length
frame_length_samples = int(fs/dF) + Lp

if Modulation_type_OFDM:
	for i in range(transmit_frames):
		QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(receive[i*frame_length_samples:(i+1)*frame_length_samples],np.ones(int(fs/dF)),symbol_length,Lp,Fc,dF)

plt.figure()

f, psd = signal.welch(receive, fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))

test = encode.QAM(0b1011, 2)
print(test)
test += complex(np.random.randn(), np.random.randn())*0.1
print(test)

print(bin(decode.QAM_nearest_neighbour(test,QAM)))

#plt.show()
