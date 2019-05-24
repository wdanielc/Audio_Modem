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
from scipy.ndimage.filters import maximum_filter1d

'''
I'm not sure if this is a good test, its possible that OFDM, the cyclic prefix,
and channel noise and response need to be simulated.

I think this will be hard to test until more demodulation stuff is added
'''

Lp = 16
dF = 100
Fc = 10500
Fs = 44000
noise_pow = 0
h_length = 300
buffer_length = 100
symbol_length = 64

h_test = np.random.randn(h_length)
h_test = h_test * (h_length-np.arange(h_length))
h_test = h_test/np.std(h_test)
h_test = 1
L = int(Fs/(2*dF))

s = np.zeros(1000)

plt.figure

for i in range(1):
    np.random.seed()
    transmit = encode.Synch_prefix(symbol_length,Lp,Fc,Fs,dF)
    transmit = transmit * 1000
    buffer = np.zeros(buffer_length)
    transmit = np.concatenate((buffer,transmit,buffer))
    receive = channel.isi_channel(transmit,1,h_test,True)
    receive = decode.time_demodulate(receive,Fs,Fc)
    
    #test = receive[Lp:Lp+int(Fs/dF)]
    #test2 = np.fft.fft(test)
    #plt.figure()
    #plt.plot(np.abs(test2))
    
    P = decode.Synch_P(receive,L)
    R = decode.Synch_R(receive,L)
#    R = decode.Synch_R(receive,int(L*1.3))
#    d = int((len(P)-len(R))/2)
#    P = P[d:len(R)+d]
#    R = R[:len(P)]
#    g = gaussian(51,10)
#    g = g/np.sum(g)
#    R = np.convolve(R,g,'same')
    R = maximum_filter1d(R,300)
    T = ((np.abs(P))**2)/(R**2)
#    plt.plot(np.log(np.abs(P)))
#    plt.plot(np.log(R))
#    plt.plot(np.log(T))
    plt.plot(T)
    plt.show()
    
    #print(len(receive))
    #print(buffer_length-np.argmax(T)-Lp)
    s[i] = buffer_length-np.argmax(T)
    
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