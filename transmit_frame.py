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
#import time
import pyaudio as pa
import wave
from config import *
import shelve
import ldpc_functions


# Fs = 44000
# dF = 16
# QAM = 1
# symbol_length = 1024
# Lp = 350
# Fc = 10050
#
# volume = 1.0


# This is a list of QAM values of the data

frame_length_bits = symbol_length*2*QAM
#data_bits = np.random.randint(2,size=frame_length_bits*100)	#generate random sequence of length = 10 frame
data_bits = data.get_data(filename)#[:1000*frame_length_bits]
#data_bits = np.ones(frame_length_bits*500, dtype=int)
with open("start_bits.txt", 'w') as fout:
	for value in data_bits:
		fout.write(str(value) + '\n')
code_bits = ldpc_functions.encode(data_bits, standard = '802.16', rate = '2/3',  ptype='A' )
transmit_frames = int(np.ceil(len(code_bits)/frame_length_bits))
frame_length_samples = int(Fs/dF) + Lp

QAM_values = data.modulate(code_bits, QAM, frame_length_bits)


transmit = np.zeros(transmit_frames * frame_length_samples, dtype = np.float32)

file = shelve.open('SNR')
SNR = file['SNR']
B = file['B']

for i in range(transmit_frames):
	waterfilled_QAM = encode.waterfilling(QAM_values[i * symbol_length:(i + 1) * symbol_length], SNR, B, dF, Fc, symbol_length)
	#waterfilled_QAM = QAM_values[i * symbol_length:(i + 1) * symbol_length]
	transmit[i * frame_length_samples: (i+1) * frame_length_samples] = encode.OFDM(waterfilled_QAM, Lp, Fc, Fs, dF)


transmit = np.insert(transmit,0,encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF))
# Truncate to remove spikes from the signal
lim = np.std(np.abs(transmit)) * 3
transmit = np.clip(transmit,-lim,lim)
transmit = transmit/max(abs(transmit))
#print(max(np.abs(transmit)), min(np.abs(transmit)), np.mean(np.abs(transmit)), np.std(np.abs(transmit)))


with open("transmit_frame.txt", 'w') as fout:
	for value in transmit:
		fout.write(str(value) + '\n')


audio.play(transmit, volume, Fs)
plt.show()