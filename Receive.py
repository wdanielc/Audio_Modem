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

if Modulation_type_OFDM:
	symbol_length = OFDM_symbol_length
else:
	symbol_length = DMT_symbol_length

#transmitted_QAM = data.modulate(data.get_data('hamlet_semiabridged.txt'), QAM, OFDM_symbol_length*2*QAM)

samples = []
recorder_state = False
record_buffer_length = 1000 # recording buffer length


def callback(in_data, frame_count, time_info, status):
	global samples, recorder_state

	mysamples = np.frombuffer(in_data, dtype=np.float32, count=frame_count)

	if recorder_state:
		samples.extend(mysamples)

	return(in_data, pa.paContinue) # returning is compulsory even in playback mode


p = pa.PyAudio()
stream = p.open(format=pa.paFloat32, channels=1, rate=fs, 
               stream_callback=callback, input=True, 
               frames_per_buffer=record_buffer_length)
stream.start_stream()


input('Press enter when ready to record')
recorder_state = True
input('Press enter to finish recording')
recorder_state = False

stream.stop_stream()
stream.close()
p.terminate()

#samples_demod = decode.time_demodulate(samples,Fs,Fc) 
#synch_metric = decode.Synch_getstart(samples_demod,int(symbol_length/2))

#assume we get the signal start here
sigstart = 0
print(frame_length)

estimation_frame = samples[sigstart + frame_length + Lp:sigstart + 2*frame_length + Lp]

print(len(estimation_frame))
gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)

time_data = samples[sigstart + 2*frame_length + Lp:]

frame_length_bits = symbol_length*2*QAM
transmit_frames = int(np.ceil(len(time_data)/frame_length_bits))

#time_data = np.append(time_data,np.zeros(frame_length - len(time_data)%frame_length))

QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length
frame_length_samples = frame_length_bits + Lp

if Modulation_type_OFDM:
	for i in range(transmit_frames):
		QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],np.ones(symbol_length),symbol_length,Lp,Fc,dF)
#
#plt.figure()
#
#f, psd = signal.welch(receive, fs, nperseg=1024)
#plt.plot(f, 20*np.log10(psd))

data_out = data_output.demodulate(QAM_values, QAM)
#print(type(data_out[0]))
#print(data_bits[:100])
data_output.write_data(data_bits)
#
##plt.show()
