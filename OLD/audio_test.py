import audio_functions as audio
import numpy as np

interval = 1.0 # interval between impulses (in seconds)
pulsewidth = 10 # width of a pulse (in samples)
# (you can try width 1 but that's extremely low power...)

# samples preparation
impulse = pulsewidth*[1]
impulse.extend((int(44100*interval)-pulsewidth)*[0])
impulses = 8*impulse
# now convert to a numpy array with the correct data type
impulses = np.array(impulses,dtype=np.float32)
# play sample
audio.play(impulses, 1.0, 44100)