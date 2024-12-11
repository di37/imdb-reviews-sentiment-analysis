import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def split_data(self, train_frac=0.7, val_frac=0.15, test_frac=0.15):
        train_df, temp_df = train_test_split(
            self.df, test_size=1 - train_frac, random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_frac / (test_frac + val_frac), random_state=42
        )
        return train_df, val_df, test_df

    def encode_labels(self, train_df, test_df, val_df, target_feature):
        labels = train_df[target_feature]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

        train_df["label"] = label_encoder.transform(train_df[target_feature])
        test_df["label"] = label_encoder.transform(test_df[target_feature])
        val_df["label"] = label_encoder.transform(val_df[target_feature])

        train_df.drop(columns=[target_feature], inplace=True)
        test_df.drop(columns=[target_feature], inplace=True)
        val_df.drop(columns=[target_feature], inplace=True)

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        return train_df, test_df, val_df

    def convert_series_to_nparray(
        self, train_df, test_df, val_df, input_feature, target_feature
    ):
        train_sentences = train_df[input_feature].to_numpy()
        test_sentences = test_df[input_feature].to_numpy()
        val_sentences = val_df[input_feature].to_numpy()

        train_labels = train_df[target_feature].to_numpy()
        test_labels = test_df[target_feature].to_numpy()
        val_labels = val_df[target_feature].to_numpy()
        return (
            train_sentences,
            test_sentences,
            val_sentences,
            train_labels,
            test_labels,
            val_labels,
        )
