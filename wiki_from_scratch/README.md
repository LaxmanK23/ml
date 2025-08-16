# wiki_from_scratch

Scripts and data for Wikipedia-based QA, pretraining, and tokenization.

## Contents

- `ask_own.py`, `finetune_qa.py`, etc.: Python scripts for QA and model training.
- `data/`: Contains Wikipedia corpus, passages, and QA data.
- `mlm_ckpt/`: Model checkpoints and tokenizer files.
- `tok/`, `tok_wp/`: Tokenizer models and vocabularies.

## Usage

1. Pretrain or finetune models:
   ```
   python finetune_qa.py
   ```
2. Generate synthetic QA:
   ```
   python make_synthetic_qa.py
   ```

See each script for specific arguments and options.
