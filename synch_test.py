#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:08:35 2019

@author: wdc24
"""

import encoding_functions as encode
import decoding_functions as decode
import numpy as np
import matplotlib.pyplot as plt

'''
I'm not sure if this is a good test, its possible that OFDM, the cyclic prefix,
and channel noise and response need to be simulated.

I think this will be hard to test until more demodulation stuff is added
'''

symbol_length = 16
noise_pow = 0
Lp = 400

randbits1 = np.random.randint(0,4,size=2*symbol_length)
randbits2 = np.random.randint(0,4,size=2*symbol_length)
randQAM1 = np.zeros(symbol_length,dtype=complex)
randQAM2 = np.zeros(symbol_length,dtype=complex)
for i in range(symbol_length):
    randQAM2[i] = encode.QAM(randbits2[i],1)
    if i % 2 == 0:
        randQAM1[i] = encode.QAM(randbits1[i],1)

test1 = encode.OFDM(randQAM1,Lp,10000,44100,10)
transmit_length = len(test1)
buffer = np.zeros(transmit_length)
test1 = np.concatenate((buffer,test1,buffer))
test1 = test1 + noise_pow * ( 0.1*np.random.randn(len(test1)) + 1j*np.random.randn(len(test1)) )
L = int((transmit_length-Lp)/2)
P1 = decode.Synch_P(test1,L)
R1 = decode.Synch_R(test1,L)
P2 = np.absolute(P1)
T = P2 / R1
plt.figure()
plt.plot(T)
#plt.plot(P2)
#plt.plot(R1)