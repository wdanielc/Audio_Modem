"""Assign variables here to ensure consistency over transmit and receive"""

# General variables
volume = 1.0 # Playback volume

filename = "hamlet.txt"
# Modulation_type_OFDM = True  #True for OFDM, False for DMT
Fs = 44000
dF = 10
QAM = 1
symbol_length = 1024
Lp = 350
#Fc = 10050
frame_length = int(Fs/dF)
frame_length_bits = symbol_length*2*QAM
frame_length_samples = frame_length + Lp

# OFDM variables
# OFDM_Fs = 44100
# OFDM_symbol_length = 1024
Fc_desired = 10000
# frame_length = int(Fs/dF)
#
# # DMT variables
# DMT_fs = 20000
# DMT_symbol_length = int(((DMT_fs/dF)-2)/2)

#Synch variables
Synch_seed = 12345678

# finding Fc closest to the desired value (Fc must be compatible with dF and symbol_length)
offset = (symbol_length%2)*0.5 - 0.5
Fc = int(offset*dF + dF*round((Fc_desired/dF)-offset))
