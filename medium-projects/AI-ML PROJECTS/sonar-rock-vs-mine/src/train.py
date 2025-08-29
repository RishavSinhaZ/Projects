# src/train.py
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_sonar(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(p, header=None)   # Sonar standard format has no header
    return df

def train_and_save(data_path, model_path, report_path, test_size=0.2, seed=42):
    start = time.time()
    print("Loading data:", data_path)
    df = load_sonar(data_path)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns (features + label)")

    X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values
    y = df.iloc[:, -1].astype(str).values

    if np.isnan(X).any():
        raise ValueError("Found non-numeric values in feature columns. Clean the CSV (remove header).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Classification report:\n")
        f.write(report)

    elapsed = time.time() - start
    print(f"Model saved: {model_path}")
    print(f"Report saved: {report_path}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Time: {elapsed:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sonar.csv", help="Path to sonar CSV")
    parser.add_argument("--model", default="models/sonar_model.pkl", help="Where to save model")
    parser.add_argument("--report", default="models/metrics.txt", help="Where to save metrics")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_save(args.data, args.model, args.report, test_size=args.test_size, seed=args.seed)
