import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/chunks/chunks.jsonl")
OUTPUT_FILE = Path("data/embeddings/embedded_chunks.jsonl")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32


# ============================================================
# Load Chunks
# ============================================================

def load_chunks(file_path):
    chunks = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
                chunks.append(chunk)

            except json.JSONDecodeError as e:
                print(
                    f"Warning: Invalid JSON on line {line_number}: {e}"
                )

    return chunks


# ============================================================
# Generate Embeddings
# ============================================================

def generate_embeddings(chunks, model):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# Save Embedded Chunks
# ============================================================

def save_embeddings(chunks, embeddings, output_file):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_file, "w", encoding="utf-8") as f:

        for chunk, embedding in zip(chunks, embeddings):

            record = {
                **chunk,
                "embedding": embedding.tolist()
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Embedding Generation Pipeline")
    print("=" * 60)

    # Check input
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Load chunks
    print("\nLoading chunks...")

    chunks = load_chunks(INPUT_FILE)

    print(f"Loaded {len(chunks)} chunks.")

    if not chunks:
        raise ValueError(
            "No chunks found in the input file."
        )

    # Load model
    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded successfully.")

    # Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(
        chunks,
        model
    )

    print(
        f"Generated embeddings with shape: {embeddings.shape}"
    )

    # Save
    print("\nSaving embedded chunks...")

    save_embeddings(
        chunks,
        embeddings,
        OUTPUT_FILE
    )

    print(f"\nSaved to: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("Embedding generation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()