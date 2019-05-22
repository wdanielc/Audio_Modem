"""Assign variables here to ensure consistency over transmit and receive"""

# General variables
volume = 1.0 # Playback volume

file = "hamlet_semiabridged.txt"
Modulation_type_OFDM = True  #True for OFDM, False for DMT
dF = 10
T = 1/dF
QAM = 2
Lp = 500
Fs = 44100

# OFDM variables
OFDM_Fs = 44100
OFDM_symbol_length = 1024
Fc = 10000
frame_length = int(Fs/dF)

# DMT variables
DMT_fs = 20000
DMT_symbol_length = int(((DMT_fs/dF)-2)/2)

#Synch variables
Synch_seed = 12345678


