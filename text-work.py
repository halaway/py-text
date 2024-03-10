import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



"""
Using Reccurrent Neural Networks, and Long Short Term 
Memory to efficiently summarize english text.
"""

class TextSummary:

    ##
    # 
    # Initializing Model 
    # #
    def __init__(self, file_name):
        try:
            self.text_data = pd.read_csv(file_name=file_name)
        except:
            print("Error loading data...")


    ##
    # Tokenizing and Parsing Training Data
    # #
    def tokenize_data(self):

        self.tokenized_entries = self.text_data




def main():


    return 0;


if __name__ == "__main__":
    main();