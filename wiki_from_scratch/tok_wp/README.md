# tok_wp

WordPiece tokenizer vocabulary for Wikipedia corpus.

- `vocab.txt`: WordPiece vocabulary file.

## How to Run

1. Install dependencies from the parent folder:

   ```
   pip install -r ../../requirements.txt
   ```

2. Run tokenization scripts (from parent folder):
   ```
   python ../train_tokenizer_wordpiece.py
   ```

This will use the vocabulary in `tok_wp/vocab.txt` for WordPiece tokenization tasks.
