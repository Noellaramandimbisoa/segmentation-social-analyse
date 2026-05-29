import joblib
import pandas as pd

def load_artifacts(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_clusters(df, model, scaler):
    X = df.drop(columns=["id"], errors="ignore")
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)