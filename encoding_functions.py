import numpy as np

class Constellation():
    pass


def bits2values():
    pass


def OFDM(symbol,):
    
    pass

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