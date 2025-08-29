# src/predict.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

def predict_from_file(model_path: str, file_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError("Model not found. Train first.")
    model = load(model_path)
    df = pd.read_csv(file_path, header=None)
    # if label included, drop it
    if df.shape[1] == 61:
        df = df.iloc[:, :-1]
    if df.shape[1] != 60:
        raise ValueError(f"Expected 60 features per row, got {df.shape[1]}")
    X = df.astype(float).values
    preds = model.predict(X)
    mapping = {"R": "ROCK", "M": "MINE"}
    for i, p in enumerate(preds):
        print(f"Sample {i}: {mapping.get(p,p)}")

def predict_from_values(model_path: str, values_str: str):
    if not Path(model_path).exists():
        raise FileNotFoundError("Model not found. Train first.")
    vals = [float(x.strip()) for x in values_str.split(",") if x.strip()!=""]
    if len(vals) != 60:
        raise ValueError(f"Provide exactly 60 values; provided {len(vals)}")
    X = np.array(vals).reshape(1, -1)
    model = load(model_path)
    p = model.predict(X)[0]
    mapping = {"R": "ROCK", "M": "MINE"}
    print("Prediction:", mapping.get(p,p))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sonar_model.pkl", help="Path to model")
    parser.add_argument("--file", help="CSV file for batch predict (60 features per row)")
    parser.add_argument("--values", help="Single-sample: 60 comma-separated values")
    args = parser.parse_args()

    if args.file:
        predict_from_file(args.model, args.file)
    elif args.values:
        predict_from_values(args.model, args.values)
    else:
        print("Provide --file <csv> OR --values \"v1,v2,...,v60\"")
