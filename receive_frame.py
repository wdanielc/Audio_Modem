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
dF = 100
QAM = 1
symbol_length = 128
Lp = 350
Fc = 10500
frame_length = int(Fs/dF)
frame_length_bits = symbol_length*2*QAM
frame_length_samples = frame_length + Lp


h_length = 100
h = [np.exp(-2*i/h_length) for i in range(h_length)]


samples = channel.get_received(sigma=0.01, h = h, ISI=True, file = "transmit_frame.txt")


#samples = []
#recorder_state = False
#record_buffer_length = 1000 # recording buffer length
#
#
#def callback(in_data, frame_count, time_info, status):
#	global samples, recorder_state
#
#	mysamples = np.frombuffer(in_data, dtype=np.float32, count=frame_count)
#
#	if recorder_state:
#		samples.extend(mysamples)
#
#	return(in_data, pa.paContinue) # returning is compulsory even in playback mode
#
#
#p = pa.PyAudio()
#stream = p.open(format=pa.paFloat32, channels=1, rate=Fs,
#               stream_callback=callback, input=True, 
#               frames_per_buffer=record_buffer_length)
#stream.start_stream()
#
#
#input('Press enter when ready to record')
#recorder_state = True
#input('Press enter to finish recording')
#recorder_state = False
#
#stream.stop_stream()
#stream.close()
#p.terminate()


samples_demod = decode.time_demodulate(samples,Fs,Fc) 							#find start of first synch block
sigstart = decode.Synch_framestart(samples_demod, int(frame_length/2))
#print(sigstart)
#sigstart = 300 + Lp

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

estimation_frame = samples[sigstart + frame_length:sigstart + 2*frame_length + Lp]						#slice out second synch block
gains = decode.get_gains(estimation_frame,encode.randQAM(symbol_length)[1],symbol_length,Lp,Fc,dF)		#get gains from second synch block
print(gains[:10])


time_data = samples[sigstart + 2*frame_length + Lp:]
time_data = np.append(time_data,np.zeros((frame_length+Lp) - ((len(time_data)-1)%(frame_length+Lp)+1)))		#- + 1 is so remainder 0 does not cause any appending
transmit_frames = int(len(time_data)/(frame_length+Lp))


QAM_values = np.zeros((transmit_frames*symbol_length), dtype = np.complex)	#initialises QAM value vector of correct length


for i in range(transmit_frames):
	QAM_values[i*symbol_length:(i+1)*symbol_length] = decode.OFDM(time_data[i*frame_length_samples:(i+1)*frame_length_samples],gains,symbol_length,Lp,Fc,dF)


data_out = data_output.demodulate(QAM_values, QAM)

#we know we only sent 10 frames
data_out = data_out[:(2*QAM*symbol_length)*100]


with open("start_bits.txt", 'r') as fin:
	transmitted = np.array(fin.read().split('\n'))

if len(transmitted) < len(data_out):
    data_out = data_out[:len(transmitted)]
elif len(transmitted) > len(data_out):
    data_out = np.concatenate((data_out,np.zeros((len(transmitted)-len(data_out)))))
    
transmitted = np.delete(transmitted, -1)
transmitted = np.array(transmitted, dtype = int)
print(len(data_out), len(transmitted))
data_out = np.array(data_out, dtype = int)#[:len(transmitted)]

errors = np.bitwise_xor(transmitted,data_out)
print(errors[:50])
print('Error rate = ',np.sum(errors)/len(data_out))

data_output.write_data(data_out, "receive_frame.txt")




