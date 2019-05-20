import numpy as np

def isi_channel(input, sigma=0.0, h=True, ISI=True):
    if h & ISI:
        import impulse_estimation as estimate
        h = estimate.get_h()

    output = input
    if ISI:
        output = np.convolve(input, h,'same')

    output += sigma * np.random.randn(len(output))
    return output

def get_received():
    with open("transmit.txt", 'r') as fin:
        transmitted = np.array(fin.read().split('\n'))

    transmitted = np.delete(transmitted, -1)
    transmitted = np.array(transmitted, dtype=np.float64)

    return isi_channel(transmitted, 0.3)

get_received()