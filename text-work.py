import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


import tensorflow as tf
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


    ##
    # 
    # Creating IDS Using Example Texts
    # #
    def create_ids(self):

        # Converting Strings to Numerical Representation
        example_texts = ['abcdefg', 'xyz']
        chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

        self._chars = chars




def main():


    # Downloading Example Shakespeare Text
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Reading File from Path
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    
    # First 250 characters
    # print(text[:250])

    # Unique Characters
    vocab = sorted(set(text))

    # Converting Strings to numerical Representations
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

    # Creating IDs
    ids_from_chars = tf.keras.layers.StringLookup(\
        vocabulary=list(vocab), mask_token=None)

    chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    ids = ids_from_chars(chars)

    chars = chars_from_ids(ids)

    l = tf.strings.reduce_join(chars, axis=-1).numpy()

    print(l)

   # print(chars)




    #print(ids)
    

    return 0;


if __name__ == "__main__":
    main();
