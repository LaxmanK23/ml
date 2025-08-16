# scrape_wiki.py
import wikipedia, re, json, os, random
from tqdm import tqdm

random.seed(42)
wikipedia.set_lang("en")

SEEDS = [
    "science", "technology", "history", "geography", "mathematics",
    "economics", "space", "biology", "medicine", "sports", "politics",
    "art", "music", "india", "world war", "ai", "machine learning"
]
MAX_PAGES = 800
CHUNK_MIN = 200
CHUNK_MAX = 1200

def clean(text: str) -> str:
    text = re.sub(r"==.*?==+", " ", text)       # drop headings
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_paragraphs(text: str):
    parts = re.split(r"(?<=\.)\s{1,}", text)    # split on sentence-ish boundaries
    para, out = "", []
    for s in parts:
        if len(para) + len(s) + 1 <= CHUNK_MAX:
            para += (" " if para else "") + s
        else:
            if len(para) >= CHUNK_MIN: out.append(para)
            para = s
    if len(para) >= CHUNK_MIN: out.append(para)
    return out

def main():
    seen, pages = set(), []
    for q in SEEDS:
        for t in wikipedia.search(q, results=50):
            if t in seen: continue
            try:
                p = wikipedia.page(t, auto_suggest=False)
                pages.append((p.title, p.url, clean(p.content)))
                seen.add(t)
                if len(pages) >= MAX_PAGES: break
            except Exception:
                continue
        if len(pages) >= MAX_PAGES: break

    os.makedirs("data", exist_ok=True)
    with open("data/wiki_corpus.txt", "w", encoding="utf-8") as fw, \
         open("data/wiki_passages.jsonl", "w", encoding="utf-8") as fp:
        for title, url, content in tqdm(pages, desc="chunking"):
            for para in split_paragraphs(content):
                fw.write(para + "\n")
                fp.write(json.dumps({"title": title, "url": url, "context": para}, ensure_ascii=False) + "\n")

    print(f"Wrote corpus to data/wiki_corpus.txt and passages to data/wiki_passages.jsonl")

if __name__ == "__main__":
    main()
