# ask_own.py
import pickle, re, argparse, torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from rank_bm25 import BM25Okapi

def tokenize(t): return re.findall(r"[A-Za-z0-9]+", t.lower())

bm = pickle.load(open("bm25.pkl", "rb"))
bm25, DOCS, TITLES, URLS = bm["bm25"], bm["docs"], bm["titles"], bm["urls"]

tok = BertTokenizerFast.from_pretrained("qa_ckpt")
model = BertForQuestionAnswering.from_pretrained("qa_ckpt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def best_context(question, k=6):
    scores = bm25.get_scores(tokenize(question))
    top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(DOCS[i], TITLES[i], URLS[i]) for i, _ in top]

def answer(question, contexts):
    best = None
    for ctx, title, url in contexts:
        inputs = tok(question, ctx, return_tensors="pt", truncation=True, max_length=384).to(device)
        with torch.no_grad():
            out = model(**inputs)
            start = out.start_logits.argmax(-1).item()
            end   = out.end_logits.argmax(-1).item()
        ans = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(inputs["input_ids"][0][start:end+1]))
        score = (out.start_logits[0, start].item() + out.end_logits[0, end].item())
        cand = {"answer": ans.strip(), "score": score, "title": title, "url": url}
        if not best or cand["score"] > best["score"]:
            best = cand
    return best

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args()

    ctxs = best_context(args.question, k=args.k)
    pred = answer(args.question, ctxs)
    print("\nQ:", args.question)
    print("A:", pred["answer"])
    print(f"score={pred['score']:.2f}  source={pred['title']}  {pred['url']}")
