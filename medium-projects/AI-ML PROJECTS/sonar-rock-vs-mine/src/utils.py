import pandas as pd
from pathlib import Path

def load_sonar_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the sonar dataset.
    Assumes last column is label ('R' or 'M'), rest are numeric features.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Some datasets come without header
    df = pd.read_csv(p, header=None)
    return df
