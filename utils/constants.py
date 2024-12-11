import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

VOCAB_SIZE = 20000  # Only consider the top 20k words
MAX_LEN = 200  # Only consider the first 200 words of each movie review

EMBED_DIM = 32  # Embedding size for each token
NUM_HEADS = 2  # Number of attention heads
FF_DIM = 32  # Hidden layer size in feed forward network inside transformer

ACTIVATION_RELU = "relu"
ACTIVATION_SOFTMAX = "softmax"

MODEL_NAME = "custom_model"
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}.pkl")
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = "accuracy"
BATCH_SIZE = 64
EPOCHS = 2

DATA_PATH = os.path.join("data", "movie_dataset.csv")
TEXT_VECTORIZER_PATH = os.path.join("models", "text_vectorizer.pkl")