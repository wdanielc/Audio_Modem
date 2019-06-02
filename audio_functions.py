import pyaudio as pa
import wave
import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt

def play(samples, volume, fs):
    p = pa.PyAudio()
    stream = p.open(format=pa.paFloat32, channels=1, rate=fs, output=True)
    stream.write(volume*np.tile(samples,4))
    stream.stop_stream()
    stream.close()
    p.terminate()


def record(duration, fs, buffer_length):
    p = pa.PyAudio()
    stream = p.open(format=pa.paFloat32, channels=1, rate=fs, input=True, frames_per_buffer = buffer_length)
    samples = []
    for i in range(0, int(fs / buffer_length * duration)):
        data = stream.read(buffer_length)
        samples.extend(np.frombuffer(data, dtype=np.float32))
    stream.stop_stream()
    stream.close()
    p.terminate()
    return(samples)

def callback(in_data, frame_count, time_info, status):
    global samples, threshold, background_power, L_init_samples, recorder_state
    
    # transfer input data to a numpy array in the right format
    mysamples = np.frombuffer(in_data, dtype=np.float32, count=frame_count)

    if recorder_state == 'acquiring':
        samples.extend(mysamples)
        if len(samples) > L_init_samples:
            background_power = np.mean(np.array(samples)**2)
            recorder_state = 'waiting'
    elif recorder_state == 'waiting':
        mypower = np.mean(mysamples**2)
        if mypower > threshold*background_power:
            samples = mysamples.tolist()
            recorder_state = 'recording'
    elif recorder_state == 'recording':
        mypower = np.mean(mysamples**2)
        if mypower < threshold*background_power:
            recorder_state = 'completed'
        else:
            samples.extend(mysamples)
    return(in_data, pa.paContinue) # returning is compulsory even in playback mode"""