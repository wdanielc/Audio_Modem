import encoding_functions as encode
import decoding_functions as decode
#import audio_functions as audio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import data_input
import data_output
import wave
import channel
#import pyaudio as pa
from scipy.ndimage.filters import maximum_filter1d


Fs = 44000
dF = 1000
QAM = 1
symbol_length = 16
Lp = 8
Fc = 10500
frame_length = int(Fs/dF)
frame_length_bits = symbol_length*2*QAM
frame_length_samples = frame_length + Lp


h_length = 1
h = [np.exp(-2*i/h_length) for i in range(h_length)]
print(h)


samples = channel.get_received(sigma=0, h = h, ISI=True, file = "transmit_frame.txt")


samples_demod = decode.time_demodulate(samples,Fs,Fc) 							#find start of first synch block
sigstart = decode.Synch_framestart(samples_demod, int(frame_length/2))
print(sigstart)
sigstart = 300 + Lp


'''P = decode.Synch_P(samples_demod, int(frame_length/2))
R = decode.Synch_R(samples_demod, int(frame_length/2))
R = maximum_filter1d(R,300)
M = ((np.abs(P))**2)/(R**2)
plt.plot(M, label = 'M')
plt.plot(abs(P), label = 'P')
plt.plot(R, label = 'R')
plt.plot(samples, label = 'signal')
plt.gca().legend()
plt.show()'''


estimation_frame = samples[sigstart + frame_length:sigstart + 2*frame_length + Lp]						#slice out second synch block
gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)		#get gains from second synch block


time_data = samples[sigstart + 2*frame_length + Lp:]
time_data = np.append(time_data,np.zeros((frame_length+Lp) - ((len(time_data)-1)%(frame_length+Lp)+1)))		#- + 1 is so remainder 0 does not cause any appending
transmit_frames = int(len(time_data)/(frame_length+Lp))


QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length


for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],gains,symbol_length,Lp,Fc,dF)


data_out = data_output.demodulate(QAM_values, QAM)

#we know we only sent one frame
data_out = data_out[:(2*QAM*symbol_length)]


with open("start_bits.txt", 'r') as fin:
	transmitted = np.array(fin.read().split('\n'))


transmitted = np.delete(transmitted, -1)
transmitted = np.array(transmitted, dtype = int)
data_out = np.array(data_out, dtype = int)

print(data_out)
print(transmitted)

errors = np.bitwise_xor(transmitted,data_out)
print('Error rate = ',np.sum(errors)/len(data_out))




