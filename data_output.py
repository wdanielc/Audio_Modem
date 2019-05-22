import decoding_functions as decode
import numpy as np
from data_input import bits2ints

def write_data(bits, file="received.txt"):

    with open(file, 'wb') as fout:
        for byte in bits2bytes(bits):
            fout.write(bytes(byte))

    return None


def demodulate(QAM_values, QAM):
    out = np.zeros((len(QAM_values) * QAM * 2), dtype=np.bool)

    decode_data = np.vectorize(decode.QAM_nearest_neighbour)
    data = decode_data(QAM_values, QAM)

    for i in range(len(data)):
        out[i * QAM * 2:(i + 1) * QAM * 2] = ints2bits(data[i], QAM)

    return np.array(out, dtype=np.int64)


def ints2bits(ints, QAM):
    return np.array([(2**(2 * QAM - 1) >> i) & ints for i in range(QAM * 2)])


# There is a problem with this which is messing up writing the file
def bits2bytes(bits):
    s = "".join(chr(int("".join(map(str, bits[i:i + 8])), 2)) for i in
                range(0, len(bits), 8))
    return s

"""
def bits2bytes(x):
    n = len(x)+3
    r = (8 - n % 8) % 8
    prefix = format(r, '03b')
    x = ''.join(str(a) for a in x)
    suffix = '0'*r
    x = prefix + x + suffix
    x = [x[k:k+8] for k in range(0,len(x),8)]
    y = []
    for a in x:
        np.append(y, a)

    return y
"""

if __name__ == "__main__":
    print(ints2bits(1, 2))
