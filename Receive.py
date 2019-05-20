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


receive = channel.get_received(sigma=0.0, h=True, ISI=False)


if Modulation_type_OFDM:
	symbol_length = OFDM_symbol_length
else:
	symbol_length = DMT_symbol_length



QAM_values = np.zeros(int((len(receive)*symbol_length)/((fs/dF) + Lp)), dtype = np.complex)	#initialises QAM value vector of correct length


if Modulation_type_OFDM:
	for i in range(int(len(QAM_values)/symbol_length)):
		QAM_values[int(i*symbol_length):int((i+1)*symbol_length)] = decode.OFDM(receive[int(i*((fs/dF) + Lp)):int((i+1)*((fs/dF) + Lp))],np.ones(int(fs/dF)),symbol_length,Lp,Fc,fs,dF)

print(QAM_values)

plt.figure()

f, psd = signal.welch(receive, fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))

plt.show()




