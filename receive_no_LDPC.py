import encoding_functions as encode
import decoding_no_LDPC as decode
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


samples_demod = decode.time_demodulate(samples,Fs,Fc) 							#find start of first synch block
# sigstart, freq_offset = decode.Synch_framestart(samples_demod, int(frame_length/2))
# sigstart = sigstart-Lp
# print(sigstart)
synch_centres = decode.get_synch_times(samples_demod,int(frame_length/2))
synch_centres = np.subtract(synch_centres, Lp)


P = decode.Synch_P(samples_demod, int(frame_length/2))
R = decode.Synch_R(samples_demod, int(frame_length/2))
R = maximum_filter1d(R,300)
M = ((np.abs(P))**2)/(R**2)
plt.plot(M, label = 'M')
plt.scatter(synch_centres, np.ones(len(synch_centres)), marker='x')
#plt.plot(abs(P), label = 'P')
#plt.plot(R, label = 'R')
# plt.plot(samples, label = 'signal')
plt.gca().legend()
plt.show()

block_length = 6 * frame_length_samples

transmit_frames = 4 * len(synch_centres)

QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)
print(len(QAM_values))

for i in range(len(synch_centres)):
	blocks, residuals = decode.split_samples(samples[synch_centres[i]:synch_centres[i]+block_length],0,frame_length,Lp)
	print(blocks.shape)

	estimation_frame = blocks[1]
	gains = decode.get_gains2(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Fc,dF,Fs)

	blocks = blocks[2:]
	residuals = residuals[2:]

	QAM_values_i = np.zeros((4*symbol_length), dtype = np.complex)

	for k in range(4):
		QAM_values_i[k*symbol_length:(k+1)*symbol_length] = decode.OFDM2(blocks[k],gains,symbol_length,Fc,dF,Fs,residuals[k])

	QAM_values[i*4*symbol_length:(i+1)*4*symbol_length] = QAM_values_i

# for i in range(transmit_frames):
# 	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(blocks[i],gains,symbol_length,Fc,dF,Fs)

print("before demod")
data_out = data_output.demodulate(QAM_values, QAM)
print('done demod')

with open("start_bits.txt", 'r') as fin:
	transmitted = np.array(fin.read().split('\n'))

transmitted = np.delete(transmitted, -1)

if len(transmitted) < len(data_out):
	data_out = data_out[:len(transmitted)]
elif len(transmitted) > len(data_out):
	data_out = np.concatenate((data_out,np.zeros((len(transmitted)-len(data_out)))))

transmitted = np.array(transmitted, dtype = int)
data_out = np.array(data_out, dtype = int)#[:len(transmitted)]

error_rates = np.zeros(len(transmitted))
errors = 0

for i in range(len(transmitted)):
	errors += np.bitwise_xor(transmitted[i],data_out[i])
	error_rates[i] = errors/(i+1)

plt.figure()
plt.plot(error_rates)

errors = np.bitwise_xor(transmitted,data_out)
print('Error rate = ',np.sum(errors)/len(data_out))

data_output.write_data(data_out, "received_" + filename)

# received = shelve.open("errors")
# #received['different'] = error_rates
#
# same = received['./shelve_files/same']
# different = received['different']
#
# plt.figure()
# plt.plot(same, label='Same Laptop')
# plt.plot(different, label='Different Laptops')
# plt.legend()
# plt.xlabel('Values')
# plt.ylabel('Error rate')

plt.show()
