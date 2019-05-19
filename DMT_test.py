import encoding_functions as encode
import numpy as np

x = [complex(1.0,1.0),complex(2.3,-1.0),complex(0.0,2.0),complex(-0.5,0.5),complex(1.0,1.0),complex(1.0,1.0)]

print(encode.DMT(x,1))