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
    pkl_file = open("ML_results\\eval_labels.pkl","rb")
    test_truth = pickle.load(pkl_file).tolist()
    pkl_file.close()

    pkl_file = open("ML_results\\train_labels.pkl","rb")
    train_truth = pickle.load(pkl_file).tolist()
    pkl_file.close()

    pkl_file = open("ML_results\\eval_predictions.pkl","rb")
    test_pre = pickle.load(pkl_file)['results'].tolist()
    pkl_file.close()

    pkl_file = open("ML_results\\train_predictions.pkl","rb")
    train_pre = pickle.load(pkl_file)['results'].tolist()
    pkl_file.close()

    plt.figure()
    plt.plot(train_truth)
    plt.plot(train_pre,'g*')
    plt.savefig("ML_results\\train.bmp") 

    plt.figure()
    plt.plot(test_truth)
    plt.plot(test_pre,'g*')
    plt.savefig("ML_results\\test.bmp") 
if __name__ == '__main__':
    label()
    # Preprocess raw data
    
