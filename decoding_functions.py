import numpy as np
from encoding_functions import grey2bin
from scipy.signal import butter, lfilter

def OFDM(received,gains,symbol_length,Lp,Fc,dF):
    trans = np.array(received[Lp:])
    spectrum = np.fft.fft(trans)
    
    sigstart = (Fc/dF) - symbol_length/2 #this isnt *exactly* centred on Fc, but oh well
    sigstart = int(round(sigstart))
    sigend = sigstart + symbol_length
    
    scaled_symbol = spectrum[sigstart:sigend]	#input signal scaled by complex channel gains
    symbol = np.divide(scaled_symbol,gains[sigstart:sigend])
    return symbol

def LPF(signal,Fs,Fc):
    b,a = butter(5,Fc/Fs)
    return lfilter(b,a,signal)
    

def time_demodulate(signal,Fs,Fc):
    t = np.arange(len(signal))*(Fc/Fs)*2*np.pi
    sig_r = signal * np.cos(t)
    sig_r = LPF(sig_r,Fs,Fc)
    sig_c = signal * np.sin(t)
    sig_c = LPF(sig_c,Fs,Fc)
    return (sig_r + 1j*sig_c)

QAM_norm = [2,10,42]

def QAM_nearest_neighbour(value, num):
    value *= QAM_norm[num-1]**0.5

    # Round to the nearest odd number
    a = int(round((1+np.real(value))/2)*2)-1
    b = int(round((1+np.imag(value))/2)*2)-1

    if abs(a) > 2*num - 1:
        if a < 0:
            a = -(2*num - 1)
        else:
            a = 2*num - 1

    if abs(b) > 2 * num - 1:
        if b < 0:
            b = -(2 * num - 1)
        else:
            b = 2 * num - 1

    a = grey2bin(int((a - 1 + num ** 2) / 2))
    b = grey2bin(int((b - 1 + num ** 2) / 2))

    var = (b << num) ^ (a)

    return var


def Synch_P(signal,L):
    length = len(signal) - 2*L
    pi = np.conj(signal[:-L]) * signal[L:]
    P = np.zeros(length,dtype=complex)
    P[0] = np.sum(pi[0:L])
    for d in range(length-1):
        P[d+1] = P[d] + pi[d+L] - pi[d]
    return P

def Synch_R(signal,L):
    length = len(signal) - 2*L
    ri = np.absolute(signal)
    ri  = ri[:-L] * ri[:-L] #this is maybe ri[L:] * ri[:-L]
    R = np.zeros(length)
    R[0] = np.sum(ri[0:L])
    for d in range(length-1):
        R[d+1] = R[d] + ri[d+L] - ri[d]
    return R