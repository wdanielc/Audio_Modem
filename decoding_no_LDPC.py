import numpy as np
import encoding_functions as encode
from scipy.signal import butter, lfilter
from scipy.ndimage.filters import maximum_filter1d


def OFDM(received, gains, symbol_length, Lp, Fc, dF):
    trans = np.array(received[Lp:])
    spectrum = np.fft.fft(trans)
    sigstart = (Fc / dF) + 0.5 - symbol_length / 2
    sigstart = int(round(sigstart))
    sigend = sigstart + symbol_length

    scaled_symbol = spectrum[
                    sigstart:sigend]  # input signal scaled by complex channel gains
    symbol = np.divide(scaled_symbol, gains)
    return symbol


def LPF(signal, Fs, Fc):
    b, a = butter(5, Fc / Fs)
    return lfilter(b, a, signal)


def time_demodulate(signal, Fs, Fc):
    t = np.arange(len(signal)) * (Fc / Fs) * 2 * np.pi
    sig_r = signal * np.cos(t)
    sig_r = LPF(sig_r, Fs, Fc)
    sig_c = signal * np.sin(t)
    sig_c = LPF(sig_c, Fs, Fc)
    return (sig_r + 1j * sig_c)


QAM_norm = [2, 10, 42]


def QAM_nearest_neighbour(QAM_value, QAM):
    QAM_value *= QAM_norm[QAM - 1] ** 0.5
    a = np.real(QAM_value)
    b = np.imag(QAM_value)
    a = (a + 2 ** QAM) / 2
    a = np.clip(int(np.floor(a)), 0, (2 ** QAM) - 1)
    b = np.clip(int(np.floor((b + 2 ** QAM) / 2)), 0, (2 ** QAM) - 1)
    a = encode.grey2bin(a)
    b = encode.grey2bin(b)
    var = (b << QAM) ^ (a)
    return var


def distance_to_Likelihood(d,
                           sigma2):  # calculate pmf of d for 0 mean gaussian with var sigma2, ignore constant scaling value as will be same number on top and bottom
    a = np.exp(-(d ** 2) / (2 * sigma2))
    return a


def QAM_LLR(QAM_value, QAM,
            sigma2):  # sigma2 corresponds to noise variance in QAM plane
    Binaries = np.arange((2 * QAM) ** 2)
    bin2QAM = np.vectorize(encode.QAM)
    Constellation = bin2QAM(Binaries, QAM)
    Distance = np.zeros(len(Binaries))
    for i in range(len(
            Binaries)):  # calculate distances to each point in constellation
        Distance[i] = abs(QAM_value - Constellation[i])
    distances_to_Likelihoods = np.vectorize(distance_to_Likelihood)
    Likelihoods = distances_to_Likelihoods(Distance, sigma2)
    LLRs = np.zeros(2 * QAM)
    for i in range(2 * QAM):
        zeros = 0
        ones = 0
        for k in range(len(Binaries)):
            if ((Binaries[k] >> ((2 * QAM - 1) - i)) & 1) == 1:
                ones += Likelihoods[k]
            else:
                zeros += Likelihoods[k]
        zeros = np.log(zeros)
        ones = np.log(ones)
        LLRs[i] = zeros - ones
    return (LLRs)


def Synch_P(signal, L):
    length = len(signal) - 2 * L
    pi = np.conj(signal[:-L]) * signal[L:]
    P = np.zeros(length, dtype=complex)
    P[0] = np.sum(pi[0:L])
    for d in range(length - 1):
        P[d + 1] = P[d] + pi[d + L] - pi[d]
    return P


def Synch_R(signal, L):
    length = len(signal) - 2 * L
    ri = np.absolute(signal)
    ri = ri[L:] * ri[:-L]  # this is maybe ri[L:] * ri[:-L]
    R = np.zeros(length)
    R[0] = np.sum(ri[0:L])
    for d in range(length - 1):
        R[d + 1] = R[d] + ri[d + L] - ri[d]
    return R


def longest_block(list_in, allowed_zeros=0, find_all=False):
    l = len(list_in)
    b_start = []
    b_end = []
    b_length = []  # this is NOT end - start)
    for i in range(l):
        if (i == 0) and (list_in[i] == 1):
            c = True
        elif (list_in[i - 1] == 0) and (list_in[i] == 1):
            c = True
        else:
            c = False
        if c:
            this_end = i;
            this_length = 1;
            this_zeros = 0
            for j in range(i + 1, l):
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
    if not find_all:
        i = np.argmax(b_length);
        return b_start[i], b_end[i], b_length[i]

    else:
        centres = []
        for i in range(len(b_length)):
            if b_length[i] > 100:
                centres.append(int((b_end[i] - b_start[i])/2))
        return centres




def Synch_framestart(signal, L, spread=300, threshold=0.8):
    P = Synch_P(signal, L)
    R = Synch_R(signal, L)
    R = maximum_filter1d(R, spread)
    M = ((np.abs(P)) ** 2) / (R ** 2)
    start, end = longest_block((M > threshold), 0)[:2]
    frame_start = int((start + end) / 2)
    freq_offset = np.mean(np.angle(P[start:end]))
    return frame_start, freq_offset


def get_synch_times(signal, L, spread=300, threshold=0.8):
    P = Synch_P(signal, L)
    R = Synch_R(signal, L)
    R = maximum_filter1d(R, spread)
    M = ((np.abs(P)) ** 2) / (R ** 2)
    return longest_block((M>threshold), 0,True)

def get_gains(estimation_frame, sent_spectrum, symbol_length, Lp, Fc,
              dF):
    estimate_spectrum = OFDM(estimation_frame, np.ones(symbol_length),
                             symbol_length, Lp, Fc, dF)

    return np.divide(estimate_spectrum, sent_spectrum)


def split_samples(signal, freq_offset, frame_length, Lp):
    frame_length_samples = frame_length + Lp
    # sample_shift_per_frame = (frame_length_samples/frame_length)*(freq_offset/np.pi)
    shifted_frame_length = frame_length_samples  # + sample_shift_per_frame
    n = int(np.ceil(len(signal) / shifted_frame_length))
    signal = np.append(signal,
                       np.zeros(int(np.floor(shifted_frame_length))))
    frame = 0
    frames = np.zeros([n, frame_length])
    residual = np.zeros(n)
    for i in range(n):
        frame = frame + shifted_frame_length
        frame_end = int(np.floor(frame))
        frame_start = frame_end - frame_length
        frames[i, :] = signal[frame_start:frame_end]
        residual[i] = (frame - frame_end)
    return frames, residual


def OFDM2(received, gains, symbol_length, Fc, dF, Fs, residual):
    spectrum = np.fft.fft(received, int(Fs / dF))

    phase_shifts = np.arange(len(spectrum))
    phase_shifts = phase_shifts * (
                (residual * 2 * np.pi) / (Fs * len(spectrum)))
    phase_shifts = np.exp(1j * phase_shifts)
    spectrum = spectrum * phase_shifts

    sigstart = (Fc / dF) + 0.5 - symbol_length / 2
    sigstart = int(round(sigstart))
    sigend = sigstart + symbol_length
    scaled_symbol = spectrum[
                    sigstart:sigend]  # input signal scaled by complex channel gains
    symbol = np.divide(scaled_symbol, gains)
    return symbol


def get_gains2(estimation_frame, sent_spectrum, symbol_length, Fc, dF,
               Fs):
    estimate_spectrum = OFDM2(estimation_frame, np.ones(symbol_length),
                              symbol_length, Fc, dF, Fs, 0)

    return np.divide(estimate_spectrum, sent_spectrum)
