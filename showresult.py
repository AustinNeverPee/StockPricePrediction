"""Data Preprocessing
Preprocess raw price data into the form of TP Matrix
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

# Settings of TP Matrix
# Both are based on Fibonacci numbers
# Range of price changes ratio


def label():
    pkl_file = open("ML_result/" + stock_name + "/eval_labels.pkl", "rb")
    test_truth = pickle.load(pkl_file).tolist()
    pkl_file.close()

    pkl_file = open("ML_result/" + stock_name + "/train_labels.pkl", "rb")
    train_truth = pickle.load(pkl_file).tolist()
    pkl_file.close()

    pkl_file = open("ML_result/" + stock_name + "/eval_predictions.pkl", "rb")
    test_pre = pickle.load(pkl_file)['results'].tolist()
    pkl_file.close()

    pkl_file = open("ML_result/" + stock_name + "/train_predictions.pkl", "rb")
    train_pre = pickle.load(pkl_file)['results'].tolist()
    pkl_file.close()

    plt.figure()
    plt.plot(train_truth)
    plt.plot(train_pre, 'g*')
    plt.savefig("ML_result/" + stock_name + "/train.png")
    plt.show()

    plt.figure()
    plt.plot(test_truth)
    plt.plot(test_pre, 'g*')
    plt.savefig("ML_result/" + stock_name + "/test.png")
    plt.show()


if __name__ == '__main__':
    # Sotck name
    stock_name = "BA"

    label()
    # Preprocess raw data
