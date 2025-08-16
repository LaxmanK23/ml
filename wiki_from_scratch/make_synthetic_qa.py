# make_synthetic_qa.py
import json, re, random
from tqdm import tqdm

random.seed(42)
IN = "data/wiki_passages.jsonl"
OUT = "data/wiki_qa.jsonl"

ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")  # naive title-case entity

def make_question(sent, answer):
    # very simple template; you can add more templates
    return f"In the following sentence, what is the missing entity? '{sent.replace(answer, '_____')}'"

def main():
    out = open(OUT, "w", encoding="utf-8")
    with open(IN, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="building QA"):
            row = json.loads(line)
            ctx = row["context"]
            sents = re.split(r"(?<=\.)\s+", ctx)
            for s in sents:
                if len(s) < 40: continue
                ents = [m.group(1) for m in ENTITY_RE.finditer(s)]
                if not ents: continue
                ans = random.choice(ents)
                start = ctx.find(ans)
                if start == -1: continue
                qa = {
                    "id": f"{row['title']}_{start}",
                    "title": row["title"],
                    "context": ctx,
                    "question": make_question(s, ans),
                    "answers": {"text": [ans], "answer_start": [start]}
                }
                out.write(json.dumps(qa, ensure_ascii=False) + "\n")
    out.close()
    print(f"Wrote synthetic QA to {OUT}")
if __name__ == "__main__":
    main()
