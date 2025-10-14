import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_irrigation_model(path="data/irrigation.csv"):
    df = pd.read_csv(path)
    X = df.drop("irrigation", axis=1)
    y = df["irrigation"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Irrigation Model Accuracy:", accuracy_score(y_test, preds))
    return model

def predict_irrigation(model, features: dict):
    X_input = [[features["humidity"], features["rainfall"], features["temperature"]]]
    return model.predict(X_input)[0]
