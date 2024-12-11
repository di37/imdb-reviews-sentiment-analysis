import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import numpy as np
import tensorflow as tf
from sentiment_analysis import FeatureEngineering, ModelSaveLoad


class MovieSentimentPredictor:
    def __init__(self, vocab_size, max_len, model_path):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.model_path = model_path
        self.loaded_model = None
        self.text_vectorizer = None

    def load_model(self):
        self.loaded_model = ModelSaveLoad.load_model(self.model_path)
        print("Model loaded successfully.")

    def load_vectorizer(self):
        feature_engineering = FeatureEngineering(self.vocab_size, self.max_len)
        self.text_vectorizer = feature_engineering.create_text_vectorizer()
        train_sentences = np.load("data/train_sentences.npy", allow_pickle=True)
        self.text_vectorizer.adapt(train_sentences)
        print("Text vectorizer loaded and adapted.")

    def preprocess_sentence(self, sentence):
        sentence_vector = self.text_vectorizer(np.array([sentence]))
        sentence_vector = tf.keras.preprocessing.sequence.pad_sequences(
            sentence_vector, maxlen=self.max_len
        )
        return sentence_vector

    def predict_sentiment(self, sentence):
        if self.loaded_model is None:
            self.load_model()
        if self.text_vectorizer is None:
            self.load_vectorizer()

        sentence_vector = self.preprocess_sentence(sentence)
        prediction_score = np.argmax(self.loaded_model.predict(sentence_vector))

        if prediction_score == 1:
            sentiment = f"Movie is Positive - Score {prediction_score}"
        else:
            sentiment = f"Movie is Negative - Score {prediction_score}"

        return sentiment