"""Data Preprocessing
Preprocess raw price data into the form of TP Matrix
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import pickle
import pdb
import random


# Settings of TP Matrix
# Both are based on Fibonacci numbers
# Range of price changes ratio
rows = ((0, 2), (2, 5), (5, 10), (10, 18), (18, 31), (31, 52), (52, 86), (86, 141), (141, 100000),
        (-2, 0), (-5, -2), (-10, -5), (-18, -10), (-31, -18), (-52, -31), (-86, -52), (-141, -86), (-100000, -141))
# Time window
columns = ((1, 2), (3, 5), (6, 10), (11, 18), (19, 31),
           (32, 52), (53, 86), (87, 141), (142, 230)) * 2


stockname = "BA"


class Stock(object):
    def __init__(self):
        self.train = []
        self.test = []


class DataSet(object):
    def __init__(self):
        self.tp_features = []
        self.labels = 1


def preproc_data(data):
    """Preprocess raw data into TP Matrix format"""
    # Load data manually from Yahoo! finance

    # Initialize TP Matrix
    # 3-dimension: # of stock * 18 * 18
    # narray
    _TP_matrixs = np.zeros(
        (len(data.ix[stockname]) - 230, 18, 18), dtype=np.bool)
    old = data.ix[stockname]['close'][229]
    TP_matrixs = pd.Panel(_TP_matrixs, items=data.ix[stockname].index[230:])
    label = np.zeros((len(data.ix[stockname]) - 230), dtype=np.float)
    dataindex = 0
    dataset = []
    # Construct TP Matrix
    for TP_matrix in TP_matrixs.iteritems():
        # Extract raw close price of last 230 days
        # pdb.set_trace()
        tp_features = np.zeros((18, 18), dtype=np.bool)
        _list_CP = data.ix[stockname][data.ix[stockname].index <
                                      TP_matrix[0]]['close'].tolist()
        list_CP = _list_CP[len(_list_CP) - 230: len(_list_CP)]
        close = data.ix[stockname]['close'][dataindex + 230]
        label = (close - old) / old
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
                        tp_features[row][col] = True
                        break

        sample = DataSet()
        sample.tp_features = tp_features
        sample.labels = label
        dataindex += 1
        dataset.append(sample)

    filename = 'data/TP_matrix_' + stockname + '.pkl'
    output = open(filename, 'wb')
    # # Pickle dictionary using protocol 0.
    pickle.dump(TP_matrixs, output)
    output.close()
    return dataset


def cv(dataset):
    random.shuffle(dataset)
    stock = Stock()
    l = int(len(dataset) * 0.7)
    stock.train = dataset[:l]
    stock.test = dataset[l:]
    return stock


if __name__ == '__main__':
    # Preprocess raw data
    filename = "data/" + stockname + ".pkl"
    pkl_file = open(filename, "rb")
    # pkl_file = open("DataSet.pkl","rb")
    data = pickle.load(pkl_file)
    pkl_file.close()

    #
    # train_labels = np.zeros(len(data), dtype=np.float)
    # for i in range(len(data)):
    #     train_labels[i] = data[i].labels
    # print (type(train_labels),train_labels)

    # label = label(data)
    dataset = preproc_data(data)
    # data = cv(dataset)
    # tp_pkl_file = open("data\\TP_matrix.pkl","rb")
    # TP_matrixs = pickle.load(tp_pkl_file)
    # tp_pkl_file.close()

    filename = "data/data_set_" + stockname + ".pkl"
    # Store TP Matrix into pickle format
    output = open(filename, 'wb')
    # # Pickle dictionary using protocol 0.
    pickle.dump(dataset, output)
    output.close()
