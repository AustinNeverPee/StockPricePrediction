"""Download raw data and save as pickle format
"""


import pandas as pd
import pytz
from datetime import datetime
from zipline.utils.factory import load_bars_from_yahoo
import pickle
import pdb

# Load data manually from Yahoo! finance
start = datetime(2010, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
data = load_bars_from_yahoo(stocks=['AAPL'],
                            start=start,
                            end=end)

pdb.set_trace()

# Store raw data into pickle format
output = open('raw data/AAPL.pkl', 'wb')
# Pickle dictionary using protocol 0.
pickle.dump(data, output)
output.close()
