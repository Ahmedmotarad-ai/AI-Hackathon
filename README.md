# Embeddings

Pipeline to generate and validate embeddings from document chunks.

## Structure

- `data/chunks/chunks.jsonl` — input chunks (from person 3)
- `data/embeddings/embedded_chunks.jsonl` — output embedded chunks
- `src/generate_embeddings.py` — generates embeddings
- `src/validate_embeddings.py` — validates embedded chunks

## Usage

```bash
pip install -r requirements.txt
python src/generate_embeddings.py
python src/validate_embeddings.py
```
