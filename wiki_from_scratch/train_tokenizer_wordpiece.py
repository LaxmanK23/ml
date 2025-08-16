# train_tokenizer_wordpiece.py
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

CORPUS = "data/wiki_corpus.txt"
OUT = Path("tok_wp")
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = BertWordPieceTokenizer(lowercase=True, strip_accents=True)

tokenizer.train(
    files=[CORPUS],
    vocab_size=30000,
    min_frequency=2,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

# Save vocab.txt + tokenizer.json (HF-compatible folder)
tokenizer.save_model(str(OUT))
print(f"Saved WordPiece tokenizer to {OUT.resolve()} (vocab.txt, tokenizer.json)")
