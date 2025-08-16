# pretrain_mlm.py
from datasets import load_dataset
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)

# 1) Load your freshly trained WordPiece tokenizer
tok = BertTokenizerFast.from_pretrained("tok_wp")  # folder containing vocab.txt

# 2) Define a small BERT config (trainable on CPU, but still non-trivial)
config = BertConfig(
    vocab_size=len(tok),
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=6,
    intermediate_size=1536,
    max_position_embeddings=512,
    type_vocab_size=2
)

model = BertForMaskedLM(config)

# 3) Load your corpus (one paragraph per line)
dataset = load_dataset("text", data_files={"train": "data/wiki_corpus.txt"})["train"]

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(
    tokenizer=tok, mlm=True, mlm_probability=0.15
)

# 4) Train (tweak batch sizes/epochs as your machine allows)
args = TrainingArguments(
    output_dir="mlm_ckpt",
    per_device_train_batch_size=8,     # lower if you run out of RAM
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    num_train_epochs=1,                # increase later
    weight_decay=0.01,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=2,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=collator
)

trainer.train(resume_from_checkpoint=True)

# 5) Save your own pretrained checkpoint (no external weights!)
trainer.save_model("mlm_ckpt")
tok.save_pretrained("mlm_ckpt")
print("Saved MLM checkpoint to mlm_ckpt/")
