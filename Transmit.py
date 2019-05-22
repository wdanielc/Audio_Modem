#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:15:46 2019

@author: wdc24
"""
import encoding_functions as encode
#import audio_functions as audio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import data_input as data
#import time
#import pyaudio as pa
import wave

filename = "hamlet_semiabridged.txt"
Modulation_type_OFDM = True  #True for OFDM, False for DMT

volume = 1.0
OFDM_Fs = 44100
DMT_Fs = 20000


Fc = 10000 # Carrier frequency
dF = 10
T = 1/dF
QAM = 2
OFDM_symbol_length = 1024
DMT_symbol_length = int(((DMT_Fs/dF)-2)/2)
Lp = 350


if Modulation_type_OFDM:
	symbol_length = OFDM_symbol_length
	Fs = OFDM_Fs
else:
	symbol_length = DMT_symbol_length
	Fs = DMT_Fs


#transmit = np.array([])
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

#p = pa.PyAudio()


'''block = QAM_values[:symbol_length]

stream = p.open(format=pa.paFloat32,
				channels=1,
				rate=Fs, 
				output=True, 
				frames_per_buffer=int(transmit_block_length),
				stream_callback=callback
				)
stream = p.open(format=pa.paFloat32,
                channels=1,
                rate=Fs,
                output=True)'''

# This is a list of QAM values of the data
data_bits = data.get_data(filename)
frame_length_bits = symbol_length*2*QAM
transmit_frames = int(np.ceil(len(data_bits)/frame_length_bits))
frame_length_samples = int(Fs/dF) + Lp

QAM_values = data.modulate(data_bits, QAM, frame_length_bits)

print(QAM_values)

transmit = np.zeros(transmit_frames * frame_length_samples)

if Modulation_type_OFDM:
	print("Starting OFDM")
	for i in range(transmit_frames):
		'''stream.write(volume*np.tile(encode.OFDM(block, 350, Fc, Fs, dF),4))'''
		transmit[i * frame_length_samples: (i+1) * frame_length_samples] = encode.OFDM(QAM_values[i * symbol_length:(i + 1) * symbol_length], Lp, Fc, Fs, dF)
else:
	print('Starting DMT')
	for i in range(transmit_frames):
		'''stream.write(volume*np.tile(encode.OFDM(block, 350, Fc, Fs, dF),4))'''
		transmit[i * frame_length_samples: (i+1) * frame_length_samples] = encode.DMT(QAM_values[i * symbol_length:(i + 1) * symbol_length], Lp)

transmit = np.insert(transmit,0,encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF))


#transmit= np.append(transmit, np.zeros(transmit_block_length-(len(transmit) % transmit_block_length)))     #Append 0s to make transmit fit evenly into data blocks of length 2*fs/df

"""
stream.stop_stream()
stream.close()
p.terminate()
"""

plt.figure()

f, psd = signal.welch(transmit, Fs, nperseg=1024)
plt.plot(f, 20*np.log10(psd))

filename = 'myAudioFile.wav'
# save result to file
samples = transmit/max(abs(transmit))
samples = (.5*transmit+.5)*(2**14)
samples = samples.astype(np.int16)
wf = wave.open(filename, 'wb')
wf.setnchannels(1)
wf.setsampwidth(2) # 2 bytes per sample int16. If unsure, use np.dtype(np.int16).itemsize
wf.setframerate(Fs)
wf.writeframes(b''.join(samples))
wf.close()

with open("transmit.txt", 'w') as fout:
	for value in transmit:
		fout.write(str(value) + '\n')


audio.play(transmit, volume, fs)


plt.show()
