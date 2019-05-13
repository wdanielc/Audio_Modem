#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:15:46 2019

@author: wdc24
"""
import encoding_functions as encode

Fc = 10000
dF = 100
T = 1/dF 

data = b'test encode'
len(data)
symbol = []
for i in range(len(data)):
    symbol.append(encode.QAM(data[i] >> 4,2))
    symbol.append(encode.QAM(data[i] % 16,2))

transmit = encode.OFDM(symbol,2)
Ts = T/len(transmit)
transmit = encode.upconvert(transmit, Fc, Ts)