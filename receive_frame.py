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
from config import *
import shelve
import ldpc_functions


# Fs = 44000
# dF = 16
# QAM = 1
# symbol_length = 1024
# Lp = 350
# Fc = 10050
# frame_length = int(Fs/dF)
# frame_length_bits = symbol_length*2*QAM
# frame_length_samples = frame_length + Lp


#h_length = 100
#h = [np.exp(-2*i/h_length) for i in range(h_length)]


samples = channel.get_received(sigma=0.01, h = 1, ISI=True, file = "transmit_frame.txt")


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
stream = p.open(format=pa.paFloat32, channels=1, rate=Fs,
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




"""
P = decode.Synch_P(samples_demod, int(frame_length/2))
R = decode.Synch_R(samples_demod, int(frame_length/2))
R = maximum_filter1d(R,300)
M = ((np.abs(P))**2)/(R**2)
plt.plot(M, label = 'M')
#plt.plot(abs(P), label = 'P')
#plt.plot(R, label = 'R')
plt.plot(samples, label = 'signal')
plt.gca().legend()
plt.show()
"""

samples_demod = decode.time_demodulate(samples,Fs,Fc) 							#find start of first synch block
sigstart, freq_offset, edge_start = decode.Synch_framestart(samples_demod, int(frame_length/2))
sigstart = sigstart-Lp
print(sigstart)

S = decode.get_freq_offset(samples_demod, freq_offset, dF, Fs, edge_start, frame_length, Lp, np.arange(-10,10));
plt.figure()
plt.plot(S)
plt.show()

blocks, residuals = decode.split_samples(samples[sigstart:],0,frame_length,Lp)
estimation_frame = blocks[1]
estimation_residual = residuals[1]
gains = decode.get_gains2(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Fc,dF,Fs,estimation_residual)

blocks = blocks[2:]
residuals = residuals[2:]
transmit_frames = len(blocks)

QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)

for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM2(blocks[i],gains,symbol_length,Fc,dF,Fs,0)

raw_LLRs = np.zeros(len(QAM_values)*2*QAM)

file = shelve.open('SNR')
sigma2 = file['noise']

print(QAM_values[:10])
for i in range(10):
	print(0.5/abs(gains[i % symbol_length]))

for i in range(len(QAM_values)):
	raw_LLRs[i*2*QAM:(i+1)*2*QAM] = decode.QAM_LLR(QAM_values[i], QAM, 0.5/abs(gains[i % symbol_length]))

print(raw_LLRs[:10])


data_bits_out = ldpc_functions.decode(raw_LLRs, standard = '802.16', rate = '2/3',  ptype='A' )

""" OLD DEDCODING
estimation_frame = samples[sigstart + frame_length:sigstart + 2*frame_length + Lp]						#slice out second synch block
gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)		#get gains from second synch block


time_data = samples[sigstart + 2*frame_length + Lp:]
time_data = np.append(time_data,np.zeros((frame_length+Lp) - ((len(time_data)-1)%(frame_length+Lp)+1)))		#- + 1 is so remainder 0 does not cause any appending
transmit_frames = int(len(time_data)/(frame_length+Lp))


QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length


for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],gains,symbol_length,Lp,Fc,dF)


data_out = data_output.demodulate(QAM_values, QAM)
"""

#we know we only sent 10 frames
#data_out = data_out[:(2*QAM*symbol_length)*100]


with open("start_bits.txt", 'r') as fin:
	transmitted = np.array(fin.read().split('\n'))

transmitted = np.delete(transmitted, -1)

if len(transmitted) < len(data_bits_out):
	data_bits_out = data_bits_out[:len(transmitted)]
elif len(transmitted) > len(data_bits_out):
	data_bits_out = np.concatenate((data_bits_out,np.zeros((len(transmitted)-len(data_bits_out)))))

transmitted = np.array(transmitted, dtype = int)
data_bits_out = np.array(data_bits_out, dtype = int)#[:len(transmitted)]

error_rates = np.zeros(len(transmitted))
errors = 0

for i in range(len(transmitted)):
	errors += np.bitwise_xor(transmitted[i],data_bits_out[i])
	error_rates[i] = errors/(i+1)

plt.figure()
plt.plot(error_rates)

errors = np.bitwise_xor(transmitted,data_bits_out)
print('Error rate = ',np.sum(errors)/len(data_bits_out))

data_output.write_data(data_bits_out, "receive_frame.txt")

received = shelve.open("errors")
#received['different'] = error_rates

same = received['same']
different = received['different']

plt.figure()
plt.plot(same, label='Same Laptop')
plt.plot(different, label='Different Laptops')
plt.legend()
plt.xlabel('Values')
plt.ylabel('Error rate')

plt.show()




