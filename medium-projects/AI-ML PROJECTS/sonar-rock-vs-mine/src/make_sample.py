# src/make_sample.py
import pandas as pd
df = pd.read_csv("data/sonar.csv", header=None)
sample = df.iloc[0, :-1]   # first row, drop label
sample.to_frame().T.to_csv("data/sample_to_predict.csv", index=False, header=False)
print("Saved data/sample_to_predict.csv")
