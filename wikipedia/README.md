# wikipedia

This folder contains scripts and data for building and querying a Wikipedia index for QA and information retrieval tasks.

## Contents

- `ask.py`: Query the Wikipedia index for answers to questions.
- `build_index.py`: Build the Wikipedia index from raw documents.
- `wiki.index`, `wiki_docs.pkl`: Index and document files (large files, ignored in git).

## Setup & Dependencies

1. Install Python dependencies from the parent folder:
   ```
   pip install -r ../requirements.txt
   ```

## How to Build and Query the Index

### 1. Build the Wikipedia Index

Run the following command to build the index from documents:

```bash
python build_index.py
```

This will create `wiki.index` and `wiki_docs.pkl` files in the folder.

### 2. Query the Wikipedia Index

Use the following command to query the index with questions:

```bash
python ask.py
```

You may need to modify `ask.py` to specify your question or input source.

## Notes

- Large files (`wiki.index`, `wiki_docs.pkl`) are ignored in git and may need to be generated or downloaded separately.
- Ensure all dependencies are installed before running scripts.
