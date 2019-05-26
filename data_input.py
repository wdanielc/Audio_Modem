import numpy as np
import encoding_functions as encode


"""This takes a file and returns a bit stream in the form of a np array"""
def get_data(file):

    with open(file, 'rb') as fin:
        data = fin.read()

    out = np.zeros(len(data)*8)
    for i in range(len(data)):
        out[i*8:(i+1)*8] = byte2bits(data[i])

    return np.array(out, dtype=np.int64)


"""This takes a bit stream (np array) and returns an array of QAM symbols. """
def modulate(bits, QAM, frame_length_bits):
    bits_per_value = 2 * QAM
    # Append a buffer of zeros to make the bit stream the correct length
    if len(bits) % frame_length_bits == 0:
        pass
    else:
        bits = np.append(bits, np.zeros(frame_length_bits - (len(bits) % frame_length_bits)))
    # Split into blocks for each QAM value
    bits = np.array([bits[i:i+bits_per_value] for i in range(0, len(bits), bits_per_value)], dtype=np.int64)
    # Convert to ints
    bits = np.array([bits2ints(block) for block in bits])

    encode_data = np.vectorize(encode.QAM)

    return encode_data(bits,QAM)


def byte2bits(y):
    bits = np.zeros(8)

    for i in range(8):
        bits[i] = y & (1 << (7-i))

    return np.array(bits, dtype=bool)


def bits2ints(y):
    out = 0
    for bit in y:
        out = (out << 1) | bit

    return out


if __name__ == "__main__":

    print(byte2bits(16))
