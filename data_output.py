import decoding_functions as decode
import numpy as np
from data_input import bits2ints


# Takes boolean array and writes it to file
def write_data(bits, file="received.txt"):

    #data_out = bytes([int(byte) for byte in bits2bytes(bits)])

    with open(file, 'wb') as fout:
        for byte in bits2bytes(bits):
            print(int(byte), bytes(int(byte)))
            fout.write(bytes([int(byte)]))

    return None


def demodulate(QAM_values, QAM):
    out = np.zeros((len(QAM_values) * QAM * 2), dtype=np.bool)

    #decode_data = np.vectorize(decode.QAM_nearest_neighbour)   #Vectorise doesn't work for some reason - it randomly changes some of the values
    #data = decode_data(QAM_values, QAM)

    for i in range(len(QAM_values)):
        decoded_value = decode.QAM_nearest_neighbour(QAM_values[i], QAM)
        out[i * QAM * 2:(i + 1) * QAM * 2] = ints2bits(decoded_value, QAM)

    return out


def ints2bits(ints, QAM):
    return np.array([(2**(2 * QAM - 1) >> i) & ints for i in range(QAM * 2)])


# Converts an array of booleans into an array of bytes (ints)
def bits2bytes(bits):
    bytes = np.zeros(int(len(bits)/8))

    for i in range(len(bytes)):
        byte_sized_chunk = bits[i * 8:(i + 1) * 8]
        bytes[i] = bits2ints(byte_sized_chunk)

    return np.array(bytes)

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
