import numpy as np
import encoding_functions as encode


"""This takes a file and returns a bit stream in the form of a np array"""
def get_data(file):

    with open(file, 'rb') as fin:
        data = fin.read()

    return bytes2bits(data)


"""This takes a bit stream (np array) and returns an array of QAM symbols. """
def modulate(bits, QAM):
    bits_per_value = 2 * QAM
    # Append a buffer of zeros to make the bit stream the correct length
    if len(bits) % bits_per_value == 0:
        pass
    else:
        bits = np.append(bits, np.zeros(bits_per_value - (len(bits) % bits_per_value)))
    # Split into blocks for each QAM value
    bits = np.array([bits[i:i+bits_per_value] for i in range(0, len(bits), bits_per_value)], dtype=np.int64)
    # Convert to ints
    bits = np.array([bits2ints(block) for block in bits])

    encode_data = np.vectorize(encode.QAM)

    return encode_data(bits,QAM)


def bytes2bits(y):
    x = [format(a, '08b') for a in y]
    r = int(x[0][0:3],2)
    x = ''.join(x)
    x = [int(a) for a in x]
    for k in range(3):
        x.pop(0)
    for k in range(r):
        x.pop()
    return np.array(x)


def bits2ints(y):
    out = 0
    for bit in y:
        out = (out << 1) | bit

    return out


if __name__ == "__main__":
    data = get_data('test.txt')

    print(modulate(data, 2))