import pyaudio as pa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import audio_functions
import time


samples = []
recorder_state = False
N = 1000 # recording buffer length
fs = 44100


def callback(in_data, frame_count, time_info, status):
	global samples, recorder_state

	mysamples = np.frombuffer(in_data, dtype=np.float32, count=frame_count)

	if recorder_state:
		samples.extend(mysamples)

	return(in_data, pa.paContinue) # returning is compulsory even in playback mode


p = pa.PyAudio()
stream = p.open(format=pa.paFloat32, channels=1, rate=fs, 
               stream_callback=callback, input=True, 
               frames_per_buffer=N)
stream.start_stream()


input('Press enter when ready to record')
start_time = time.time()
recorder_state = True
input('Press enter to finish recording')
recorder_state = False
stop_time = time.time()


stream.stop_stream()
stream.close()
p.terminate()


plt.plot(np.linspace(0,stop_time-start_time,len(samples)).tolist(), samples)
plt.grid()
plt.show()


f, psd = signal.welch(samples, fs, nperseg=1024)    #noise is white so psd should give channel transfer function scaled by variance- possibly should normalise
plt.plot(f,10*np.log10(psd))
plt.grid()
plt.show()