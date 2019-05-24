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
from scipy.ndimage.filters import maximum_filter1d


Fs = 44000
dF = 1000
QAM = 1
symbol_length = 8
Lp = 4
Fc = 10005
frame_length = int(Fs/dF)
frame_length_bits = symbol_length*2*QAM
frame_length_samples = frame_length + Lp


samples = channel.get_received(sigma=0, h = [1], ISI=True, file = "transmit_frame.txt")
samples = np.insert(samples, 0, np.zeros(1000))


#samples_demod = decode.time_demodulate(samples,Fs,Fc) 							#find start of first synch block
#sigstart = decode.Synch_framestart(samples_demod, int(frame_length/2))
sigstart = 1000 + Lp


estimation_frame = samples[sigstart + frame_length:sigstart + 2*frame_length + Lp]						#slice out second synch block
gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)		#get gains from second synch block


time_data = samples[sigstart + 2*frame_length + Lp:]
time_data = np.append(time_data,np.zeros((frame_length+Lp) - ((len(time_data)-1)%(frame_length+Lp)+1)))		#- + 1 is so remainder 0 does not cause any appending
transmit_frames = int(len(time_data)/(frame_length+Lp))


QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length


for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],gains,symbol_length,Lp,Fc,dF)


data_out = data_output.demodulate(QAM_values, QAM)
print(np.array(data_out, dtype = int))



