import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import pickle


class ModelSaveLoad:
    def __init__(self, model):
        self.model = model

    def save_model(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)
    
    @staticmethod
    def load_model(file_path):
        with open(file_path, "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model
