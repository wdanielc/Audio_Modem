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
import time
import pyaudio as pa

filename = "hamlet.txt"

volume = 1.0
fs = 44100


Fc = 10000 # Carrier frequency
dF = 10
T = 1/dF
QAM = 2
symbol_length = int(((fs/dF) -2)/2)


transmit = np.array([])
'''transmit_block_length = int(8*fs/dF)    #length of blocks of audio data to be buffered at a time, 8 OFDM symbols seems to be good balance of loop performance and not having too many callbacks
transmit_location = 0    #how far through the transmit vector has been played so far


def callback(in_data, frame_count, time_info, status): #callback function to continuously buffer and play audio
	global  transmit, transmit_location
	while len(transmit) < frame_count:
		print('pass1')
		pass
	data = transmit[transmit_location:transmit_location+frame_count]
	print(data)
	transmit_location += frame_count
	return(data, pa.paContinue)'''


# This is a list of QAM values of the data
QAM_values = data.modulate(data.get_data(filename), QAM)

print(len(QAM_values))

QAM_values = np.append(QAM_values, np.zeros(symbol_length - len(QAM_values) % symbol_length))

print(QAM_values)

#transmit = np.array([])

print(len(QAM_values)/symbol_length)
print(int(len(QAM_values)/symbol_length))


p = pa.PyAudio()


'''block = QAM_values[:symbol_length]

stream = p.open(format=pa.paFloat32,
				channels=1,
				rate=fs, 
				output=True, 
				frames_per_buffer=int(transmit_block_length),
				stream_callback=callback
				)'''
'''stream = p.open(format=pa.paFloat32,
                channels=1,
                rate=fs,
                output=True)'''



for i in range(0,int(len(QAM_values)/symbol_length)):
	block = np.array(QAM_values[i * symbol_length:(i + 1) * symbol_length])
	'''stream.write(volume*np.tile(encode.OFDM(block, 350, Fc, fs, dF),4))'''
	transmit = np.append(transmit, encode.DMT(block, 350))


'''transmit= np.append(transmit, np.zeros(transmit_block_length-(len(transmit) % transmit_block_length)))     #Append 0s to make transmit fit evenly into data blocks of length 2*fs/df


while stream.is_active():
	time.sleep(0.1)'''


'''stream.stop_stream()
stream.close()
p.terminate()'''


plt.figure()
f, psd = signal.welch(transmit, fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))

print(transmit)


audio.play(transmit, volume, fs)


plt.show()
