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
rows = ((0, 2), (2, 5), (5, 10), (10, 18), (18, 31), (31, 52), (52, 86), (86, 141), (141, 100000),
        (-2, 0), (-5, -2), (-10, -5), (-18, -10), (-31, -18), (-52, -31), (-86, -52), (-141, -86), (-100000, -141))
# Time window
columns = ((1, 2), (3, 5), (6, 10), (11, 18), (19, 31),
           (32, 52), (53, 86), (87, 141), (142, 230)) * 2

class DataSet(object):
    def __init__(self):
        self.tf = []
        self.labels = [] 

def preproc_data(data):
    """Preprocess raw data into TP Matrix format"""
    # Load data manually from Yahoo! finance

    # Initialize TP Matrix
    # 3-dimension: # of stock * 18 * 18
    # narray
    _TP_matrixs = np.zeros((len(data.ix['AAPL']) - 230 - 1, 18, 18), dtype=np.bool)
    old = data.ix['AAPL']['close'][230]
    TP_matrixs = pd.Panel(_TP_matrixs, items=data.ix['AAPL'].index[230:-1])
    label = np.zeros((len(data.ix['AAPL']) - 230),dtype=np.float)
    dataindex = 0
    # Construct TP Matrix
    for TP_matrix in TP_matrixs.iteritems():
        # Extract raw close price of last 230 days
        _list_CP = data.ix['AAPL'][data.ix['AAPL'].index < TP_matrix[0]]['close'].tolist()
        list_CP = _list_CP[len(_list_CP) - 230: len(_list_CP)]
        close = data.ix['AAPL']['close'][dataindex+231]
        label[dataindex] = (close-old)/old
        old = close
        # col[0, 8] for Upward TP Matrix
        # col[9, 17] for Downward TP Matrix
        for col in range(0, 18):
            D = columns[col][0] - 1
            for row in range(0, 18):
                # For each element of TP Matrix
                for TP in range(D, columns[col][1]):
                    # Change ratio of stock on day D with repect to the price
                    # at TP
                    C_TPD = (list_CP[TP] - list_CP[D]) / list_CP[D]
                    if C_TPD * 100 >= rows[row][0] and C_TPD * 100 < rows[row][1]:
                        TP_matrix[1][row][col] = True  
                        _TP_matrixs[dataindex][row][col] = True
                        break
        dataindex += 1
    dataset = DataSet()
    dataset.tf = _TP_matrixs
    dataset.labels = label
    return dataset

def label(data):
    AAPL = data.ix['AAPL']['close'].tolist()
    label = []
    old = 0
    for close in AAPL:
        label.append((close-old)/close)
        old = close;
    return np.array(label)

if __name__ == '__main__':
    # Preprocess raw data

    pkl_file = open("data\\AAPL.pkl","rb")
    data = pickle.load(pkl_file)
    pkl_file.close()

    #label = label(data)
    dataset = preproc_data(data)
    # tp_pkl_file = open("data\\TP_matrix.pkl","rb")
    # TP_matrixs = pickle.load(tp_pkl_file)
    # tp_pkl_file.close()

    # Store TP Matrix into pickle format
    # output = open('TP_matrix.pkl', 'wb')
    # # Pickle dictionary using protocol 0.
    # pickle.dump(TP_matrixs, output)
    # output.close()
