import numpy as np

def isi_channel(input, sigma, h, ISI):
#    if h and ISI:
#        import impulse_estimation as estimate
#        h = estimate.get_h()

    output = input
    if ISI:
        output = np.convolve(input, h,'same')

    output += sigma * np.random.randn(len(output))
    return output

def get_received(sigma=0.0, h=True, ISI=True, file = "transmit.txt"):
    with open(file, 'r') as fin:
        transmitted = np.array(fin.read().split('\n'))

    transmitted = np.delete(transmitted, -1)
    transmitted = np.array(transmitted, dtype=np.float64)

    transmitted = np.insert(transmitted, 0, np.zeros(1000)) #adds delay- can be made random later

    return isi_channel(transmitted, sigma, h, ISI)
