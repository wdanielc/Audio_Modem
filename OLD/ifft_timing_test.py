#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:21:28 2019

@author: wdc24
"""

import numpy as np
import time

reps1 = 1000
reps2 = 10000

tic1 = time.clock()
for i in range(reps):
    test = np.random.randn(44100) + 1j * np.random.randn(44100)
toc1 = time.clock()
tic2 = time.clock()
for i in range(reps):
    test = np.random.randn(44100) + 1j * np.random.randn(44100)
    test = np.fft.ifft(test)
toc2 = time.clock()
t1 = toc1-tic1
t2 = toc2-tic2
print(t2-t1)
