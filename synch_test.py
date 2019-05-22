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
import channel
from scipy.signal import gaussian
from config import *

'''
I'm not sure if this is a good test, its possible that OFDM, the cyclic prefix,
and channel noise and response need to be simulated.

I think this will be hard to test until more demodulation stuff is added
'''

#Lp = 350
#dF = 10
#Fc = 10005
#Fs = 44100
#symbol_length = 1024
noise_pow = 0
h_length = 1
buffer_length = 3000

#h_test = np.random.randn(h_length)
#h_test = h_test * (h_length-np.arange(h_length))
h_test = 0.01

s = np.zeros(1000)

for i in range(1000):
    transmit = encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF)
    transmit = transmit * 10000
    buffer = np.zeros(buffer_length)
    transmit = np.concatenate((buffer,transmit,buffer,buffer,buffer))
    receive = channel.isi_channel(transmit,0.1,h_test,True)
    receive = decode.time_demodulate(receive,Fs,Fc)
    
    #test = receive[Lp:Lp+int(Fs/dF)]
    #test2 = np.fft.fft(test)
    #plt.figure()
    #plt.plot(np.abs(test2))
    
    P = decode.Synch_P(receive,512)
    R = decode.Synch_R(receive,512)
    
    T = ((np.abs(P))**2)/(R**2)
    #g = gaussian(11,5)
    #g = g/np.sum(g)
    #T = np.convolve(g,T,'same')
    #plt.figure
    #plt.plot(np.log(T))
    
    #print(len(receive))
    #print(buffer_length-np.argmax(T)-Lp)
    s[i] = buffer_length-np.argmax(T)-Lp
    
print(np.max(s))
print(np.min(s))
print(np.sum(s)/1000)

#test = np.log(T)
#test = test>0
#print(np.sum(test))

#transmit_length = len(test1)
#buffer = np.zeros(transmit_length)
#test1 = np.concatenate((buffer,test1,buffer))
#test1 = test1 + noise_pow * ( 0.1*np.random.randn(len(test1)) + 1j*np.random.randn(len(test1)) )
#L = int((transmit_length-Lp)/2)
#P1 = decode.Synch_P(test1,L)
#R1 = decode.Synch_R(test1,L)
#P2 = np.absolute(P1)
#T = P2 / R1
#plt.figure()
#plt.plot(T)
##plt.plot(P2)
##plt.plot(R1)