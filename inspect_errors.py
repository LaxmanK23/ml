# inspect_errors.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split

CSV = r"C:\office\ml\spam.csv"  # or just "spam.csv" if you run from that folder

# Load with latin-1 and Kaggle columns
df = pd.read_csv(CSV, encoding="latin-1")
if {"v1","v2"}.issubset(df.columns):
    df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"})
df["label"] = df["label"].astype(str).str.lower().map({"ham":0,"spam":1}).astype(int)

Xtr, Xte, ytr, yte = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = joblib.load("sms_spam_model.pkl")
pred = model.predict(Xte)

print("\nFalse Positives (pred spam but actually ham):")
for t, x in zip(yte, Xte[pred==1]):
    # only those where true=0 and pred=1
    pass
for text, y_true, y_hat in zip(Xte.tolist(), yte.tolist(), pred.tolist()):
    if y_true==0 and y_hat==1:
        print("-", text[:180].replace("\n"," "), "...")

print("\nFalse Negatives (missed spam):")
for text, y_true, y_hat in zip(Xte.tolist(), yte.tolist(), pred.tolist()):
    if y_true==1 and y_hat==0:
        print("-", text[:180].replace("\n"," "), "...")
