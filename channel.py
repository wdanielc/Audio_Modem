import numpy as np

def isi_channel(input, sigma, ISI=True, noise=True):
    import impulse_estimation as estimate
    h = estimate.get_h()

    output = input
    if ISI:
        output = np.convolve(input, h,'same')
    if noise:
        output += sigma * np.random.randn(len(output))
    return output
