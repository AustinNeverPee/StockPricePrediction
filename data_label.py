"""Data Preprocessing
Preprocess raw price data into the form of TP Matrix
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import pickle


# Settings of TP Matrix
# Both are based on Fibonacci numbers
# Range of price changes ratio
def label():
    pkl_file = open("data\\AAPL.pkl","rb")
    data = pickle.load(pkl_file)
    pkl_file.close()
    AAPL_CLOSE = data.ix['AAPL']['close'].tolist()
    label = []
    old = 0
    for close in AAPL_CLOSE:
        label.append((close-old)/close)
        old = close;
    print (len(np.array(label)))
    return np.array(label)

if __name__ == '__main__':
    label()
    # Preprocess raw data
    
