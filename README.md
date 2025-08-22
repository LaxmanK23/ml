# ML Project

This repository contains code and data for machine learning experiments and Wikipedia-based NLP tasks.

## Structure

- `app.py`, `predict.py`, etc.: Main scripts for spam detection and ML tasks.
- `wiki_from_scratch/`: Scripts and data for Wikipedia-based QA and pretraining.
- `wikipedia/`: Scripts for building and querying a Wikipedia index.

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run main scripts as needed:
   ```
   python app.py
   ```

See subfolder READMEs for more details.

for Run use

```
uvicorn app:app --reload --host 127.0.0.1 --port 9000
```
