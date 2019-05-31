import ldpc_functions
import numpy as np 
import math
import decoding_functions as decode
import encoding_functions as encode
import channel


data = np.random.randint(2, size = 100)

code = ldpc_functions.encode(data, standard = '802.16', rate = '2/3',  ptype='A' )

print(code[:10])

transmit = np.zeros(len(code), dtype = np.float32)

for i in range(len(code)):
	transmit[i] = 0.1*np.random.randn() + code[i]
'''
decode = ldpc_functions.decode(code, standard = '802.16', rate = '2/3',  ptype='A' )	#this doesnt work- should take LLRs

print(data[:10])
print(decode[:10])

errors = np.bitwise_xor(data,decode)
print('Error rate = ',np.sum(errors)/len(data))


out = decode.QAM_LLR(complex(-0.9, -0.9),2,0.01)
print(out)


Binaries = np.arange((4)**2) 

print(Binaries)

for i in range(len(Binaries)):
	Binaries[i] = encode.bin2grey(Binaries[i])

print(Binaries)
'''