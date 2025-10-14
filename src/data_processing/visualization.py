import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_crop_distribution(path="../data/crop_recommendation.csv"):
    df = pd.read_csv(path)
    plt.figure(figsize=(10,5))
    sns.countplot(x="label", data=df, order=df["label"].value_counts().index)
    plt.xticks(rotation=90)
    plt.title("Crop Distribution in Dataset")
    plt.show()

def plot_correlation(path="../data/crop_recommendation.csv"):
    df = pd.read_csv(path)
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
