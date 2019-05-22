import numpy as np
from config import *

def OFDM(symbol,Lp,Fc,Fs,dF):
    spectrum = np.zeros(int(Fs/dF),dtype=complex)
    
    sigstart = (Fc/dF) - (len(symbol)/2) #this isnt *exactly* centred on Fc, but oh well
    sigstart = int(round(sigstart))
    sigend = sigstart + len(symbol)
    
    spectrum[sigstart:sigend] = symbol
    spectrum[(1-sigend):(1-sigstart)] = np.flip(np.conj(symbol))
    
    trans = np.fft.ifft(spectrum)
    trans = np.insert(trans, 0, trans[-Lp:])
    trans = np.real(trans) #ifft isnt perfect
    return trans

def DMT(symbol, Lp):      #2*len(symbol)+2 samples in freq domain = Fs*1/dF samples in time domain
    spectrum = np.append(symbol,np.conj(np.flip(symbol)))
    spectrum = np.insert(spectrum, [0,len(symbol)], 0)
    trans = np.fft.ifft(spectrum)
    trans = np.real(trans)  #ifft isnt perfect
    trans = np.insert(trans, 0, trans[-Lp:])
    return trans

QAM_norm = [2,10,42]

def QAM(var, num): #1 = 4QAM, 2=16QAM, ...
    a = var % (1 << num)
    b = var >> (num)
    a = 1 - 2**num + 2*bin2grey(a)
    b = 1 - 2**num + 2*bin2grey(b)
    return (complex(a,b)/(QAM_norm[num-1]**0.5))

def bin2grey(var):
    return(var ^ (var >> 1))

def grey2bin(var):
    mask = var >> 1
    while mask != 0:
        var = var ^ mask
        mask = mask >> 1
    return var

def randQAM(symbol_length):
    np.random.seed(Synch_seed)
    randbits1 = np.random.randint(0,4,size=2*symbol_length)
    randbits2 = np.random.randint(0,4,size=2*symbol_length)
    randQAM1 = np.zeros(symbol_length,dtype=complex)
    randQAM2 = np.zeros(symbol_length,dtype=complex)
    for i in range(symbol_length):
        randQAM2[i] = QAM(randbits2[i],1)
        if i % 2 == 0:
            randQAM1[i] = QAM(randbits1[i],1)
    return randQAM1, randQAM2

def Synch_prefix(symbol_length,Lp,Fc,Fs,dF):
    randQAM1, randQAM2 = randQAM(symbol_length)
    out1 = OFDM(randQAM1,Lp,Fc,Fs,dF)
    out2 = OFDM(randQAM2,Lp,Fc,Fs,dF)
    return np.concatenate((out1,out2))
    