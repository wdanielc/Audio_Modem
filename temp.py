def get_noisevar(estimation_frame, sent_spectrum,symbol_length,Fc,dF,Fs, gains):
    estimate_spectrum = OFDM2(estimation_frame, np.ones(symbol_length), symbol_length,Fc,dF,Fs,0)
    scaled_estimate_spectrum = np.divide(estimate_spectrum, gains)
    print(estimate_spectrum,sent_spectrum)
    var = np.square(np.absolute(np.subtract(estimate_spectrum, sent_spectrum)))
    return var


blocks, residuals = decode.split_samples(samples[sigstart:],0,frame_length,Lp)
estimation_frame = blocks[1:6]
gains = np.zeros(symbol_length)

for i in range(5):
	gains = gains + decode.get_gains2(estimation_frame[i],encode.randQAM(symbol_length)[1],symbol_length,Fc,dF,Fs)
gains = np.divide(gains,5)

noisevar = np.zeros(symbol_length)

for i in range(5):
	noisevar = noisevar + decode.get_noisevar(estimation_frame[i],encode.randQAM(symbol_length)[1],symbol_length,Fc,dF,Fs, gains)
noisevar = np.divide(noisevar, 5)

print('noisevar')
print(noisevar[:10])


blocks = blocks[6:]
residuals = residuals[6:]
transmit_frames = len(blocks)