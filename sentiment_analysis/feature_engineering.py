import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import pickle


class FeatureEngineering:
    def __init__(self, max_vocab_length, max_length):
        self.max_vocab_length = max_vocab_length
        self.max_length = max_length

    def create_text_vectorizer(self):
        """
        Create a text vectorizer with specified parameters and return it
        """
        text_vectorizer = TextVectorization(
            max_tokens=self.max_vocab_length,
            output_mode="int",
            output_sequence_length=self.max_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
        )
        return text_vectorizer