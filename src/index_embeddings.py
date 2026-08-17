import json
from pathlib import Path

import chromadb


INPUT_FILE = Path("data/embeddings/embedded_chunks.jsonl")
DB_PATH = "data/vector_db"
COLLECTION_NAME = "medical_guidelines"

EXPECTED_DIMENSION = 384


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                )

    return records


def prepare_metadata(record):
    metadata = {
        "document": str(record["document"]),
        "section": str(record["section"]),
        "page": int(record["page"]),
        "page_start": int(record["page_start"]),
        "page_end": int(record["page_end"]),
    }

    if "section_path" in record:
        metadata["section_path"] = " > ".join(
            str(x) for x in record["section_path"]
        )

    return metadata


def main():
    print("=" * 60)
    print("Vector Database Indexing")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("\nLoading embedded chunks...")

    records = load_records(INPUT_FILE)

    print(f"Loaded {len(records)} records.")

    if not records:
        raise ValueError("No records found.")

    # Validate embeddings before indexing
    for record in records:
        embedding = record.get("embedding")

        if not isinstance(embedding, list):
            raise ValueError(
                f"Invalid embedding for {record.get('chunk_id')}"
            )

        if len(embedding) != EXPECTED_DIMENSION:
            raise ValueError(
                f"Wrong dimension for {record.get('chunk_id')}: "
                f"{len(embedding)}"
            )

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    ids = [
        record["chunk_id"]
        for record in records
    ]

    documents = [
        record["text"]
        for record in records
    ]

    embeddings = [
        record["embedding"]
        for record in records
    ]

    metadatas = [
        prepare_metadata(record)
        for record in records
    ]

    print("\nIndexing embeddings...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("\n" + "=" * 60)
    print("INDEXING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Records indexed: {collection.count()}")
    print(f"Embedding dimension: {EXPECTED_DIMENSION}")
    print("Distance metric: cosine")
    print(f"Database path: {DB_PATH}")


if __name__ == "__main__":
    main()