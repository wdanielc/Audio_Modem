import decoding_functions as decode
import numpy as np
import pyaudio as pa
import matplotlib.pyplot as plt
from config import *

symbol_length = 1024
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
print(sigstart)

samples = samples[sigstart:sigstart + 2 * (int(Fs/dF) + Lp)]

frame1 = decode.OFDM(samples[:int(Fs/dF) + Lp], np.ones(symbol_length), symbol_length,Lp,Fc,dF)
frame2 = decode.OFDM(samples[int(Fs/dF) + Lp:], np.ones(symbol_length), symbol_length,Lp,Fc,dF)

signal = (frame1 + frame2) / 2
noise = sum(abs(frame1 - frame2)) / (len(frame1))

SNR = signal / (noise * np.ones(len(frame1)))

plt.plot(SNR)

plt.show()
