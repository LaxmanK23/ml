# ask.py
import pickle, faiss, numpy as np, argparse, textwrap
from transformers import pipeline

K = 6  # how many chunks to retrieve

def load_index():
    index = faiss.read_index("wiki.index")
    with open("wiki_docs.pkl", "rb") as f:
        store = pickle.load(f)
    return index, store["docs"], store["metas"]

def embed_query(q):
    # To keep things light, re-use the same embedding model name as build step:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    v = m.encode([q], normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

def retrieve(index, docs, metas, qvec, k=K):
    sims, idx = index.search(qvec, k)
    idx = idx[0].tolist()
    return [(docs[i], metas[i], float(sims[0][j])) for j, i in enumerate(idx)]

def answer(question, contexts):
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    # concatenate top contexts; long contexts increase latency
    context = "\n".join([c[:800] for c, _, _ in contexts])[:3000]
    out = qa(question=question, context=context)
    return out, context

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", type=str, help="Your question")
    p.add_argument("--k", type=int, default=K, help="Top-k passages to use")
    args = p.parse_args()

    index, docs, metas = load_index()
    qvec = embed_query(args.question)
    hits = retrieve(index, docs, metas, qvec, k=args.k)

    pred, used_ctx = answer(args.question, hits)
    print("\nQ:", args.question)
    print("A:", pred.get("answer"), f"(score={pred.get('score'):.3f})")
    print("\nTop sources:")
    for (chunk, meta, sim) in hits:
        t = meta["title"]
        u = meta["url"]
        print(f"- {t}  (sim={sim:.3f})  {u}")
