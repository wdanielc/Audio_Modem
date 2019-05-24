import numpy as np
from encoding_functions import grey2bin
from scipy.signal import butter, lfilter
from scipy.ndimage.filters import maximum_filter1d

def OFDM(received,gains,symbol_length,Lp,Fc,dF):
    trans = np.array(received[Lp:])
    spectrum = np.fft.fft(trans)
    sigstart = (Fc/dF) + 0.5 - symbol_length/2
    sigstart = int(round(sigstart))
    sigend = sigstart + symbol_length
    
    scaled_symbol = spectrum[sigstart:sigend]	#input signal scaled by complex channel gains
    symbol = np.divide(scaled_symbol,gains)
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

def QAM_nearest_neighbour(QAM_value, QAM):
    QAM_value *= QAM_norm[QAM-1]**0.5

    '''a = int(round((1+np.real(QAM_value))/2)*2)-1
    b = int(round((1+np.imag(QAM_value))/2)*2)-1


    if abs(a) > 2*QAM - 1:
        if a < 0:
            a = -(2*QAM - 1)
        else:
            a = 2*QAM - 1

    if abs(b) > 2 *QAM - 1:
        if b < 0:
            b = -(2 *QAM - 1)
        else:
            b = 2 *QAM- 1'''


    a = 0 
    b = 0


    for i in range(1,QAM+1):                    #for each bit in the grey code: check if positive or negative, set 1 if positive, then rescale value for positive/negative split to be centered for next bit
        if np.real(QAM_value) > 0:
            a += (1 << (QAM - i))
            QAM_value -= (QAM - i)
        else:
            QAM_value += (QAM -i)
        if np.imag(QAM_value) > 0:
            b += (1 << (QAM - i))
            QAM_value -= (QAM - i)
        else:
            QAM_value += (QAM -i)


    '''a = grey2bin(int((a - 1 + QAM ** 2) / 2))
    b = grey2bin(int((b - 1 + QAM ** 2) / 2))'''

    var = (b << QAM) ^ (a)
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
    ri  = ri[L:] * ri[:-L] #this is maybe ri[L:] * ri[:-L]
    R = np.zeros(length)
    R[0] = np.sum(ri[0:L])
    for d in range(length-1):
        R[d+1] = R[d] + ri[d+L] - ri[d]
    return R

def Synch_framestart(signal,L,spread=300):
    P = Synch_P(signal,L)
    R = Synch_R(signal,L)
    R = maximum_filter1d(R,spread)
    M = ((np.abs(P))**2)/(R**2)
    return np.argmax(M)

def get_gains(estimation_frame, sent_spectrum,symbol_length,Lp,Fc,dF):
    estimate_spectrum = OFDM(estimation_frame, np.ones(symbol_length), symbol_length,Lp,Fc,dF)

    return np.divide(estimate_spectrum, sent_spectrum)
