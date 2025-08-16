# real_text_classifier.py
# Train a real text classifier on 20 Newsgroups (space / baseball / politics)

from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1) Pick a few real categories
CATEGORIES = ["sci.space", "rec.sport.baseball", "talk.politics.misc"]

# 2) Load train/test splits (downloaded on first run)
train = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=("headers","footers","quotes"))
test  = fetch_20newsgroups(subset="test",  categories=CATEGORIES, remove=("headers","footers","quotes"))

# 3) Build a pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=2)),
    ("clf", LogisticRegression(max_iter=2000))
])

# 4) Train
model.fit(train.data, train.target)

# 5) Evaluate
pred = model.predict(test.data)
acc = accuracy_score(test.target, pred)
print(f"Accuracy: {acc:.3f}\n")

print("Classification report:")
print(classification_report(test.target, pred, target_names=CATEGORIES))

cm = confusion_matrix(test.target, pred)
print("Confusion matrix (rows=true, cols=pred):")
print(cm)

# 6) Try a few real predictions
samples = [
    "The shuttle will reach low earth orbit and deploy the satellite.",
    "The pitcher threw a nasty curveball in the ninth inning.",
    "The government passed a controversial bill after a heated debate.",
]
proba = model.predict_proba(samples)
labels = model.classes_  # numeric indices
inv_labels = {i: name for i, name in enumerate(CATEGORIES)}

print("\nSample predictions:")
for s, p, c in zip(samples, proba, model.predict(samples)):
    top_idx = np.argsort(p)[::-1][:3]
    tops = ", ".join([f"{inv_labels[i]}:{p[i]:.2f}" for i in top_idx])
    print(f"- '{s}' → {inv_labels[c]}  |  {tops}")
