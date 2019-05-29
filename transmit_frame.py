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
#from config import *


Fs = 44000
dF = 16
QAM = 1
symbol_length = 1024
Lp = 350
Fc = 10050

volume = 1.0


# This is a list of QAM values of the data

frame_length_bits = symbol_length*2*QAM
#data_bits = np.random.randint(2,size=frame_length_bits*100)	#generate random sequence of length = 10 frame
data_bits = data.get_data("hamlet_semiabridged.txt")#[:1000*frame_length_bits]
#data_bits = np.ones(frame_length_bits*500, dtype=int)
with open("start_bits.txt", 'w') as fout:
	for value in data_bits:
		fout.write(str(value) + '\n')
transmit_frames = int(np.ceil(len(data_bits)/frame_length_bits))
frame_length_samples = int(Fs/dF) + Lp

QAM_values = data.modulate(data_bits, QAM, frame_length_bits)


transmit = np.zeros(transmit_frames * frame_length_samples, dtype = np.float32)

for i in range(transmit_frames):
	#waterfilled_QAM = encode.waterfilling(QAM_values[i * symbol_length:(i + 1) * symbol_length], Nf, Hf, 1, dF, symbol_length)
	waterfilled_QAM = QAM_values[i * symbol_length:(i + 1) * symbol_length]
	transmit[i * frame_length_samples: (i+1) * frame_length_samples] = encode.OFDM(waterfilled_QAM, Lp, Fc, Fs, dF)


transmit = np.insert(transmit,0,encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF))


with open("transmit_frame.txt", 'w') as fout:
	for value in transmit:
		fout.write(str(value) + '\n')


audio.play(transmit, volume, Fs)
