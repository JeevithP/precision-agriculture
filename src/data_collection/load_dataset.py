import pandas as pd

def load_crop_data(path="../data/crop_recommendation.csv"):
    return pd.read_csv(path)
