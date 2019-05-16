import pyaudio as pa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import audio_functions
import time

fs = 44100

samples = [] # this will hold the recorded samples
# THE TRESHOLD VALUE IS HEURISTIC AND YOU MAY WANT TO PLAY WITH IT...
threshold = 15.0 # experimented with this empirically- 15 seems like a good value at least with my equipment- Andreas
background_power = 0.0 # mean power before the signal starts (to be measured)
L_init_samples = fs # number of initial samples to determine background mean power
# the recorder state variable tells the callback function in what phase it is and
# hence what it needs to do: 
#  - during the 'acquiring' phase the background (silent) mean power is measured
#  - during the 'waiting' phase the recorder waits for the power to jump above threshold
#  - during the 'recording' phase the signal is being recorded unless the signal has
#      jumped back down
#  - the 'completed' phase signals to the main loop that signal recording has terminated


recorder_state = 'acquiring'
N = 1000


rec_start_timeout = 10 # number of seconds tolerated before recording starts
rec_duration_timeout = 25 # longest duration of signal to be recorded


def callback(in_data, frame_count, time_info, status):
    global samples, threshold, background_power, L_init_samples, recorder_state
    
    # transfer input data to a numpy array in the right format
    mysamples = np.frombuffer(in_data, dtype=np.float32, count=frame_count)

    if recorder_state == 'acquiring':
        print(mysamples)
        samples.extend(mysamples)
        if len(samples) > L_init_samples:
            background_power = np.mean(np.array(samples)**2)
            recorder_state = 'waiting'
    elif recorder_state == 'waiting':
        mypower = np.mean(mysamples**2)
        if mypower > threshold*background_power:
            print(mysamples)
            samples = mysamples.tolist()
            recorder_state = 'recording'
    elif recorder_state == 'recording':
        mypower = np.mean(mysamples**2)
        if mypower < threshold*background_power:
            recorder_state = 'completed'
        else:
            samples.extend(mysamples)
    return(in_data, pa.paContinue) # returning is compulsory even in playback mode


# start pyAudio and instruct it to operate in callback mode
p = pa.PyAudio()
stream = p.open(format=pa.paFloat32, channels=1, rate=fs, 
               stream_callback=callback, input=True, 
               frames_per_buffer=N)
stream.start_stream()

# main loop: do nothing except check for timeouts
start_time = time.time()

# re-initialise the relevant globals in case you want to re-run this cell
recorder_state = 'acquiring'
samples = []
old_recorder_state = recorder_state
print('Acquiring backgroud power')
while True:
    if old_recorder_state == 'acquiring' and recorder_state == 'waiting':
        print('Waiting for signal to be detected: clap now!')
    elif recorder_state == 'waiting' and time.time() - start_time > rec_start_timeout:
        print('Waiting for signal start timed out')
        break
    elif old_recorder_state == 'waiting' and recorder_state == 'recording':
        print('Started recording signal')
        rec_start_time = time.time()
    elif recorder_state == 'recording' and time.time()-rec_start_time > rec_duration_timeout:
        print('Recording timed out')
        break
    elif recorder_state == 'completed':
        print('Completed the recording')
        break
    old_recorder_state = recorder_state       
    time.sleep(0.2)

stream.stop_stream()
stream.close()
p.terminate()

plt.plot(samples)
plt.grid()
plt.show()