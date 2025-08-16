# build_index.py
import wikipedia, re, pickle, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB, fast
CHUNK_SIZE = 800  # characters
OVERLAP = 120

def clean(text: str) -> str:
    text = re.sub(r"==.*?==+", " ", text)  # drop headings
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def fetch_pages(seed_queries, max_pages=200):
    seen, pages = set(), []
    for q in seed_queries:
        for title in wikipedia.search(q, results=30):
            if title in seen: continue
            try:
                p = wikipedia.page(title, auto_suggest=False)
                pages.append((p.title, p.url, clean(p.content)))
                seen.add(title)
                if len(pages) >= max_pages: return pages
            except Exception:
                continue
    return pages

def build(seeds):
    wikipedia.set_lang("en")
    pages = fetch_pages(seeds, max_pages=300)  # adjust as needed
    print(f"Fetched {len(pages)} pages.")
    docs, metas = [], []
    for title, url, content in pages:
        for c in chunk(content):
            if len(c) < 200: continue
            docs.append(c)
            metas.append({"title": title, "url": url})
    print(f"Total chunks: {len(docs)}")

    model = SentenceTransformer(EMB_MODEL)
    embs = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized dot
    index.add(embs)

    with open("wiki_docs.pkl", "wb") as f: pickle.dump({"docs": docs, "metas": metas}, f)
    faiss.write_index(index, "wiki.index")
    print("Saved wiki.index and wiki_docs.pkl")

if __name__ == "__main__":
    # Pick a broad seed list to cover many topics
    SEEDS = [
        "science", "technology", "history", "geography", "mathematics",
        "economics", "space", "biology", "medicine", "sports", "politics",
        "art", "music", "india", "usa", "world war", "ai", "machine learning"
    ]
    build(SEEDS)
