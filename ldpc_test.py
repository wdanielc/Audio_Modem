import ldpc_functions
import numpy as np 
import math
import encoding_functions as encode
import decoding_functions as decode
import audio_functions as audio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import data_input as data
#import time
import pyaudio as pa
import wave
#from config import *
import channel

Fs = 44000
dF = 1000
QAM = 2
symbol_length = 8
Lp = 8
Fc = 10050

variances = np.linspace(0.008,0.125,50)

RawErrorRate = np.zeros(50)
ErrorRate = np.zeros(50)

for i in range(len(variances)):
	sigma2 = variances[i]

	frame_length_bits = symbol_length*2*QAM
	frame_length_samples = int(Fs/dF) + Lp

	data_bits = np.random.randint(2, size = 20*frame_length_bits)
	code_bits = ldpc_functions.encode(data_bits, standard = '802.16', rate = '2/3',  ptype='A' )
	QAM_values = data.modulate(code_bits, QAM, frame_length_bits)



	noise = math.sqrt(sigma2)*np.random.randn(2,len(QAM_values))
	cmplx_noise = noise[0,:] + 1j * noise[1,:]

	QAM_values = np.add(QAM_values, cmplx_noise)


	raw_LLRs = np.zeros(len(QAM_values)*2*QAM)

	for k in range(len(QAM_values)):
		raw_LLRs[k*2*QAM:(k+1)*2*QAM] = decode.QAM_LLR(QAM_values[k], QAM, sigma2)

	errors = np.bitwise_xor(code_bits,np.vectorize(ldpc_functions.MLValue)(raw_LLRs))
	RawErrorRate[i] = np.sum(errors)/len(code_bits)


	data_bits_out = ldpc_functions.decode(raw_LLRs, standard = '802.16', rate = '2/3',  ptype='A' )


	errors = np.bitwise_xor(data_bits,data_bits_out)
	ErrorRate[i] = np.sum(errors)/len(data_bits_out)

plt.plot(variances,RawErrorRate, label = 'Raw error rate')
plt.plot(variances,ErrorRate, label = 'LDPC error rate')
plt.gca().legend()
plt.show()
