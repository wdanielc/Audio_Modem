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
sigstart, freq_offset = decode.Synch_framestart(samples_demod, int(frame_length/2))
sigstart = sigstart-Lp
print(sigstart)


blocks, residuals = decode.split_samples(samples[sigstart:],0,frame_length,Lp)
estimation_frame = blocks[1]
gains = decode.get_gains2(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Fc,dF,Fs)

blocks = blocks[2:]
residuals = residuals[2:]
transmit_frames = len(blocks)

QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)

for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM2(blocks[i],gains,symbol_length,Fc,dF,Fs,residuals[i])

data_out = data_output.demodulate(QAM_values, QAM)


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
