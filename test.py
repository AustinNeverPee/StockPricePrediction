import pickle


if __name__ == '__main__':
    # Preprocess raw data

    #pkl_file = open("data\\AAPL.pkl","rb")
    pkl_file = open("data/AAPL.pkl", "rb")
    data = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open("DataSet.pkl", "rb")
    data = pickle.load(pkl_file)
    pkl_file.close()
