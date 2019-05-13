import numpy as np

class Constellation():
    pass


def bits2values():
    pass


def OFDM():
    pass


def QAM(in, num): #1 = 4QAM, 2=16QAM, ...
    a = in % (1 << num)
    b = in >> (num)
    a = 1 - 2**num + 2*grey(a,num)
    b = 1 - 2**num + 2*grey(b,num)
    return (a + b*i)    #close butt no cigar

def grey(in,num): #todo
    pass