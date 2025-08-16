# train.py
# Train a spam detector: CSV with columns label (ham/spam) and text

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse
import sys

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path,encoding="latin-1")
    df = df.rename(columns={"v1":"label","v2":"text"})
    # Try to find label/text columns robustly
    cols = {c.lower(): c for c in df.columns}
    label_col = cols.get("label") or cols.get("category") or cols.get("class")
    text_col  = cols.get("text")  or cols.get("message")  or cols.get("sms")
    if not label_col or not text_col:
        raise ValueError(f"Could not find label/text columns in {csv_path}. "
                         f"Expected columns like label/text or category/message")
    df = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    # Normalize labels: ham=0, spam=1
    df["label"] = df["label"].astype(str).str.lower().map({"ham": 0, "spam": 1}).astype(int)
    df["text"] = df["text"].astype(str)
    return df

def main(args):
    csv_path = Path(args.data)
    out_path = Path(args.out)
    df = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification report (0=ham, 1=spam):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, out_path)
    print(f"\nSaved model to: {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sms.csv", help="Path to CSV (default: sms.csv)")
    parser.add_argument("--out",  default="sms_spam_model.pkl", help="Output model file")
    main(parser.parse_args())
