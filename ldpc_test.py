import ldpc_functions
import numpy as np 
import math
import decoding_functions as decode
import encoding_functions as encode


'''data = np.random.randint(2, size = 100)

code = ldpc_functions.encode(data, standard = '802.16', rate = '2/3',  ptype='A' )

decode = ldpc_functions.decode(code, standard = '802.16', rate = '2/3',  ptype='A' )	#this doesnt work- should take LLRs

print(data[:10])
print(decode[:10])

errors = np.bitwise_xor(data,decode)
print('Error rate = ',np.sum(errors)/len(data))'''

x = decode.QAM_LLR(0,1)
print(x)
QAM = np.vectorize(encode.QAM)
print(QAM(x,1))



