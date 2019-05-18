import numpy as np

class Constellation():
    pass


def bits2values():
    pass


def OFDM(symbol,Lp,Fc,Fs,dF):
    spectrum = np.zeros(int(Fs/dF),dtype=complex)
    
    sigstart = (Fc/dF) - (len(symbol)/2) #this isnt *exactly* centred on Fc, but oh well
    sigstart = int(round(sigstart))
    sigend = sigstart + len(symbol)
    
    spectrum[sigstart:sigend] = symbol
    spectrum = np.conj(spectrum[::-1]) #this is earier than working out the correct indicies to insert the mirrored symbol
    spectrum[sigstart:sigend] = symbol
    
    trans = np.fft.ifft(spectrum)
    trans = np.insert(trans, 0, trans[-Lp:])
    trans = np.real(trans) #ifft isnt perfect
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

def upconvert(symbol,Fc,Ts):
    freq = 2 * np.pi * Fc * Ts

    sin = np.sin(freq * np.arange(len(symbol)))
    cos = np.cos(freq * np.arange(len(symbol)))
    transmit = symbol.real * cos + symbol.imag * sin

    return transmit

# Interpolates so the actual playback time is the one corresponding to T (zero order hold)
def interpolate(symbol, fs, T):
    repeats = T*fs/len(symbol)+1
    return np.repeat(symbol, repeats)