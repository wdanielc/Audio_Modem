import decoding_functions as decode
import numpy as np
import audio_functions as audio
import pyaudio as pa
import matplotlib.pyplot as plt
from scipy import signal
from config import *
import shelve
import time

recorder_state = False
record_buffer_length = 1000 # recording buffer length
duration = 5


""" OLD
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

samples = samples[sigstart:sigstart + 10 * (int(Fs/dF) + Lp)]

frames = np.array([np.zeros(symbol_length) for i in range(10)])

for i in range(len(frames)):
	frames[i] = np.abs(decode.OFDM(samples[i * (int(Fs/dF) + Lp):(i + 1) * (int(Fs/dF) + Lp)], np.ones(symbol_length), symbol_length,Lp,Fc,dF))

frame1 = np.abs(decode.OFDM(samples[:int(Fs/dF) + Lp], np.ones(symbol_length), symbol_length,Lp,Fc,dF))
frame2 = np.abs(decode.OFDM(samples[int(Fs/dF) + Lp:], np.ones(symbol_length), symbol_length,Lp,Fc,dF))

# Do these need squaring or not??
signal = (sum(frames)) / len(frames)
noise = sum(abs(frames-signal))/len(frames)
print(noise)

"""

file = shelve.open("./shelve_files/SNR", writeback=True)


input('Press Enter when ready to record transmitted white noise')
signal_samples = audio.record(duration, Fs, record_buffer_length)
print('Done!')

input('Press Enter when ready to record background noise')
noise_samples = audio.record(duration, Fs, record_buffer_length)
print('Done!')

SNR = np.divide(np.abs(np.fft.fft(signal_samples)), np.abs(np.fft.fft(noise_samples)))

SNR = np.array([np.mean(SNR[i * duration: (i + 1) * duration]) for i in range(22000)])
noise_samples = np.array([np.mean(np.abs(noise_samples[i * duration: (i + 1) * duration])) for i in range(22000)])

noise = np.zeros(symbol_length)
for i in range(symbol_length):
    noise[i] = np.mean(noise_samples[(Fc - int(symbol_length*dF/2)) + i * dF:(Fc - int(symbol_length*dF/2)) + (i + 1) * dF])

file['noise'] = noise

B = np.mean(np.divide(np.ones(len(SNR)), SNR)) + np.std(np.divide(np.ones(len(SNR)), SNR)) * 3
file['B'] = B

file['SNR'] = SNR

file.close()

plt.figure()
#plt.plot(np.arange(int(len(SNR)/2))/5, SNR[:int(len(SNR)/2)])
plt.plot(np.arange(len(SNR)), np.divide(np.ones(len(SNR)), SNR), label='1/SNR')
plt.plot(np.arange(len(SNR)), B * np.ones(len(SNR)), linestyle='--', label='B')
plt.legend(loc=1)
#plt.ylim(0,100)
plt.xlabel('Frequency')
plt.savefig('./figures/waterfilling_real.png', dpi=300,bbox_inches='tight')

plt.show()
