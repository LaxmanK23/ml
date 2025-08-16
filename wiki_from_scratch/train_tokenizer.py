# train_tokenizer.py
import sentencepiece as spm, os
os.makedirs("tok", exist_ok=True)
spm.SentencePieceTrainer.Train(
    input="data/wiki_corpus.txt",
    model_prefix="tok/wiki_spm",
    vocab_size=30000,
    character_coverage=0.9995,
    model_type="unigram",
    input_sentence_size=1000000,
    shuffle_input_sentence=True
)
print("Tokenizer files in tok/wiki_spm.model & tok/wiki_spm.vocab")
