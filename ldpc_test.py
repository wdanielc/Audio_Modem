from ldpc_jossy.py import ldpc
import ldpc_functions
import numpy as np 
import math


data = np.random.randint(2, size = 100)
print(data.shape)

data = ldpc_functions.encode(data, standard = '802.16', rate = '2/3',  ptype='A' )

print(data.shape)