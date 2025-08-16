# retriever_bm25.py
import json, re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pickle

DOCS, TITLES, URLS = [], [], []

with open("data/wiki_passages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        DOCS.append(row["context"])
        TITLES.append(row["title"])
        URLS.append(row["url"])

def tokenize(t):
    # simple whitespace + punctuation split
    return re.findall(r"[A-Za-z0-9]+", t.lower())

tokenized_corpus = [tokenize(d) for d in DOCS]
bm25 = BM25Okapi(tokenized_corpus)
with open("bm25.pkl", "wb") as fw:
    pickle.dump({"bm25": bm25, "docs": DOCS, "titles": TITLES, "urls": URLS}, fw)
print("BM25 index saved to bm25.pkl")
