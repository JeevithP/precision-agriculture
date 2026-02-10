import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data_processing.south_india_calibration import calibrate_input

def train_crop_model(path="data/crop_recommendation.csv"):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Crop Model Accuracy:", accuracy_score(y_test, preds))
    return model

def predict_crop(model, input_data, district=None):
    if district:
        input_data = calibrate_input(input_data, district)

    df = pd.DataFrame([input_data])
    return model.predict(df)[0]