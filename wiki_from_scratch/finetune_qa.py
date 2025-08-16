# # finetune_qa.py
# import json, os
# from datasets import load_dataset
# from transformers import BertTokenizerFast, BertForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
# import numpy as np

# MODEL_DIR = "mlm_ckpt"   # your own pretrained weights

# tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
# model = BertForQuestionAnswering.from_pretrained(MODEL_DIR)

# # load jsonl into HF dataset
# data_files = {"train": "data/wiki_qa.jsonl"}
# raw = load_dataset("json", data_files=data_files, split="train")

# def preprocess(examples):
#     questions = [q.strip() for q in examples["question"]]
#     contexts  = examples["context"]
#     tokenized = tok(
#         questions, contexts,
#         max_length=384, truncation="only_second",
#         padding="max_length", return_offsets_mapping=True
#     )
#     start_positions, end_positions = [], []
#     for i, offsets in enumerate(tokenized["offset_mapping"]):
#         start_char = examples["answers"][i]["answer_start"][0]
#         end_char   = start_char + len(examples["answers"][i]["text"][0])
#         sequence_ids = tokenized.sequence_ids(i)

#         # find context token span
#         idx = 0
#         while sequence_ids[idx] != 1: idx += 1
#         context_start = idx
#         while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
#         context_end = idx - 1

#         # set default to CLS
#         start_tok, end_tok = 0, 0
#         for j in range(context_start, context_end + 1):
#             s, e = offsets[j]
#             if s <= start_char < e:
#                 start_tok = j
#             if s < end_char <= e:
#                 end_tok = j
#                 break
#         tokenized["offset_mapping"][i] = None
#         start_positions.append(start_tok)
#         end_positions.append(end_tok)

#     tokenized["start_positions"] = start_positions
#     tokenized["end_positions"]   = end_positions
#     return tokenized

# proc = raw.map(preprocess, batched=True, remove_columns=raw.column_names)

# args = TrainingArguments(
#     output_dir="qa_ckpt",
#     per_device_train_batch_size=8,
#     learning_rate=3e-5,
#     num_train_epochs=1,        # increase as needed
#     weight_decay=0.01,
#     logging_steps=100,
#     save_steps=2000,
#     save_total_limit=2,
#     fp16=False
# )

# trainer = Trainer(model=model, args=args, train_dataset=proc, data_collator=default_data_collator)
# trainer.train()
# trainer.save_model("qa_ckpt")
# tok.save_pretrained("qa_ckpt")
# print("Saved QA checkpoint to qa_ckpt/")



# finetune_qa.py — Windows-safe single-process pipeline
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_CACHING"] = "1"

from datasets import load_dataset
from transformers import (
    BertTokenizerFast, BertForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator
)

MODEL_DIR = "mlm_ckpt"

tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
tok.model_max_length = 512
model = BertForQuestionAnswering.from_pretrained(MODEL_DIR)

raw = load_dataset("json", data_files={"train": "data/wiki_qa.jsonl"}, split="train")

max_length = 384
doc_stride  = 64

def clamp_question(q: str, max_q_tokens: int = 64) -> str:
    ids = tok.encode(q, add_special_tokens=False)
    return tok.decode(ids[:max_q_tokens])

def preprocess(examples):
    questions = [clamp_question(q.strip()) for q in examples["question"]]
    contexts  = examples["context"]

    tokenized = tok(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offsets = tokenized.pop("offset_mapping")

    start_positions, end_positions = [], []

    for i, offset in enumerate(offsets):
        sample_idx = sample_mapping[i]
        ans = examples["answers"][sample_idx]
        start_char = ans["answer_start"][0]
        end_char   = start_char + len(ans["text"][0])

        sequence_ids = tokenized.sequence_ids(i)
        ctx_start = sequence_ids.index(1)
        ctx_end   = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offset[ctx_start][0] <= start_char and offset[ctx_end][1] >= end_char):
            start_positions.append(0); end_positions.append(0); continue

        st = ctx_start
        while st <= ctx_end and offset[st][0] <= start_char: st += 1
        st -= 1
        ed = ctx_end
        while ed >= ctx_start and offset[ed][1] >= end_char: ed -= 1
        ed += 1

        start_positions.append(st)
        end_positions.append(ed)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"]   = end_positions
    return tokenized

# IMPORTANT: no multiprocessing, no cache writes
proc = raw.map(
    preprocess,
    batched=True,
    remove_columns=raw.column_names,
    load_from_cache_file=False,
    desc="Tokenizing (single-process)"
)

args = TrainingArguments(
    output_dir="qa_ckpt",
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=2000,
    save_total_limit=2,
    fp16=False,
    dataloader_num_workers=0,          # <- single-process data loading
    remove_unused_columns=False,       # keep all features
    report_to=[]                       # no TB to keep it simple
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=proc,
    data_collator=default_data_collator
)

trainer.train()
trainer.save_model("qa_ckpt")
tok.save_pretrained("qa_ckpt")
print("Saved QA checkpoint to qa_ckpt/")
