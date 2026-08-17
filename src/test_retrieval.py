import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "data/vector_db"
COLLECTION_NAME = "medical_guidelines"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY = "What are the symptoms and signs of chronic heart failure?"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RELEVANCE_LABELS = {
    "nice_hf_2018_chunk_0014": ("Relevant", 1.0),
    "nice_hf_2018_chunk_0015": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0027": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0042": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0017": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0028": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0033": ("Partially Relevant", 0.5),
    "nice_hf_2018_chunk_0002": ("Not Relevant", 0.0),
    "nice_hf_2018_chunk_0003": ("Not Relevant", 0.0),
    "nice_hf_2018_chunk_0012": ("Not Relevant", 0.0),
    "nice_hf_2018_chunk_0035": ("Not Relevant", 0.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Evaluation Test"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=True,
        choices=[3, 5, 10],
        help="Number of top results to retrieve (3, 5, or 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    k = args.top_k

    print("=" * 60)
    print("Vector Search Test")
    print("=" * 60)
    print(f"\nTop-K: {k}")
    print(f"\nQuery:\n{QUERY}\n")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_text = QUERY_PREFIX + QUERY
    query_embedding = model.encode(
        query_text, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"section": {"$ne": "Front matter"}}
    )

    print("=" * 60)
    print(f"TOP {k} RESULTS")
    print("=" * 60)

    relevant_count = 0
    partial_count = 0
    not_relevant_count = 0
    total_score = 0.0

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        document = results["metadatas"][0][i]["document"]
        section = results["metadatas"][0][i]["section"]
        page = results["metadatas"][0][i]["page"]
        text = results["documents"][0][i]

        label, score = RELEVANCE_LABELS.get(
            chunk_id, ("Not Evaluated", 0.0)
        )

        if label == "Relevant":
            relevant_count += 1
        elif label == "Partially Relevant":
            partial_count += 1
        else:
            not_relevant_count += 1
        total_score += score

        print(f"\nRank: {i + 1}")
        print("-" * 40)
        print(f"Chunk ID: {chunk_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Document: {document}")
        print(f"Section: {section}")
        print(f"Page: {page}")
        print(f"Relevance: {label}")
        print(f"Relevance Score: {score}")
        try:
            print(f"Text Preview:\n{text[:800]}")
        except UnicodeEncodeError:
            safe = text[:800].encode("ascii", "replace").decode("ascii")
            print(f"Text Preview:\n{safe}")

    relevance_pct = (total_score / k) * 100
    precision_at_k = (relevant_count / k) * 100

    print()
    print("=" * 60)
    print(f"TOP-{k} SUMMARY")
    print("=" * 60)
    print(f"\nRetrieved:        {k}")
    print(f"Relevant:         {relevant_count}")
    print(f"Partially Relevant: {partial_count}")
    print(f"Not Relevant:     {not_relevant_count}")
    print(f"\nRelevance %:      {relevance_pct:.2f}%")
    print(f"Precision@{k}:    {precision_at_k:.2f}%")


if __name__ == "__main__":
    main()
