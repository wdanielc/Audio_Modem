import numpy as np

def OFDM(received,gains,symbol_length,Lp,Fc,Fs,dF):
	trans = np.array(received[Lp:])
	spectrum = np.fft.rfft(trans)

	sigstart = (Fc/dF) - symbol_length/2 #this isnt *exactly* centred on Fc, but oh well
    sigstart = int(round(sigstart))
    sigend = sigstart + symbol_length

    scaled_symbol = spectrum[sigstart:sigend]	#input signal scaled by complex channel gains
    symbol = np.divide(scaled_symbol,gains[sigstart:sigend])
    return symbol


