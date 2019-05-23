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
import data_input
import data_output
import wave
import channel
import pyaudio as pa
from config import *
from scipy.ndimage.filters import maximum_filter1d


filename = "hamlet_output.txt"	#file to save to
Modulation_type_OFDM = True  #True for OFDM, False for DMT


receive = channel.get_received(sigma=0.00, h=True, ISI=False)


if Modulation_type_OFDM:
	symbol_length = OFDM_symbol_length
else:
	symbol_length = DMT_symbol_length

#transmitted_QAM = data.modulate(data.get_data('hamlet_semiabridged.txt'), QAM, OFDM_symbol_length*2*QAM)

samples = []
recorder_state = False
record_buffer_length = 1000 # recording buffer length

h_length = 50

receive = channel.get_received(sigma=0, h = np.random.randn(h_length), ISI=True)

samples = np.insert(receive, 0, np.zeros(1000))

samples_demod = decode.time_demodulate(samples,Fs,Fc) 
sigstart = decode.Synch_framestart(samples_demod, int(frame_length/2), 800)
print(sigstart)
sigstart = 1500

estimation_frame = samples[sigstart + frame_length:sigstart + 2*frame_length + Lp]

gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)

time_data = samples[sigstart + 2*frame_length + Lp:]

P = decode.Synch_P(samples_demod, int(frame_length/2))
R = decode.Synch_R(samples_demod, int(frame_length/2))
R = maximum_filter1d(R,300)
M = ((np.abs(P))**2)/(R**2)
plt.plot(M)

frame_length_bits = symbol_length*2*QAM
transmit_frames = int(np.ceil(len(time_data)/(frame_length+Lp)))

time_data = np.append(time_data,np.zeros((frame_length+Lp) - len(time_data)%(frame_length+Lp)))


QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length
frame_length_samples = frame_length + Lp

if Modulation_type_OFDM:
	for i in range(transmit_frames):
		QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],gains,symbol_length,Lp,Fc,dF)
#
#plt.figure()
#
#f, psd = signal.welch(receive, fs, nperseg=1024)
#plt.plot(f, 20*np.log10(psd))

data_out = data_output.demodulate(QAM_values, QAM)
#print(type(data_out[0]))
#print(data_bits[:100])
data_output.write_data(data_out)

#
plt.show()

