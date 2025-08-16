# threshold_tune.py
import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv("spam.csv", encoding="latin-1")
if {"v1","v2"}.issubset(df.columns):
    df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"})
df["label"] = df["label"].str.lower().map({"ham":0,"spam":1}).astype(int)

Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.2,
                                      random_state=42, stratify=df["label"])
model = joblib.load("sms_spam_model.pkl")
proba = model.predict_proba(Xte)[:,1]

for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
    pred = (proba >= th).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    print(f"threshold={th:.2f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")
