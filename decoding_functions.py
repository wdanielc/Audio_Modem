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
    a = np.real(QAM_value)
    b = np.imag(QAM_value)
    a = (a + 2**QAM)/2
    a = np.clip(int(np.floor(a)),0,(2**QAM)-1)
    b = np.clip(int(np.floor((b + 2**QAM)/2)),0,(2**QAM)-1)
    a = grey2bin(a)
    b = grey2bin(b)
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

def longest_block(list_in,allowed_zeros=0):
    l = len(list_in)
    b_start = []
    b_end = []
    b_length = [] #this is NOT end - start)
    for i in range(l):
        if (i == 0) and (list_in[i] == 1):
            c = True
        elif (list_in[i-1] == 0) and (list_in[i] == 1):
            c = True
        else:
            c = False
        if c:
            this_end = i;
            this_length = 1;
            this_zeros = 0
            for j in range(i+1,l):
                if list_in[j] == 1:
                    this_length += 1
                    this_end = j
                else:
                    this_zeros += 1
                if this_zeros > allowed_zeros:
                    break
            b_start.append(i)
            b_end.append(this_end)
            b_length.append(this_length)
    i = np.argmax(b_length);
    return b_start[i], b_end[i], b_length[i]

def Synch_framestart(signal,L,spread=300,threshold=0.8):
    P = Synch_P(signal,L)
    R = Synch_R(signal,L)
    R = maximum_filter1d(R,spread)
    M = ((np.abs(P))**2)/(R**2)
    start, end = longest_block((M>threshold),0)
    frame_start = int((start+end)/2)
    freq_offset = np.angle(P(frame_start))
    return frame_start, freq_offset

def get_gains(estimation_frame, sent_spectrum,symbol_length,Lp,Fc,dF):
    estimate_spectrum = OFDM(estimation_frame, np.ones(symbol_length), symbol_length,Lp,Fc,dF)

    return np.divide(estimate_spectrum, sent_spectrum)
