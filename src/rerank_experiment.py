import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


DB_PATH = "data/vector_db"
COLLECTION_NAME = "medical_guidelines"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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


def evaluate(candidates, k, label=""):
    print(f"\n{'=' * 60}")
    print(f"TOP-{k} {label}")
    print(f"{'=' * 60}")

    total_score = 0.0
    relevant_count = 0
    partial_count = 0
    not_relevant_count = 0

    for i, c in enumerate(candidates[:k]):
        rel_label, score = RELEVANCE_LABELS.get(
            c["chunk_id"], ("Not Evaluated", 0.0)
        )
        if rel_label == "Relevant":
            relevant_count += 1
        elif rel_label == "Partially Relevant":
            partial_count += 1
        else:
            not_relevant_count += 1
        total_score += score

        print(f"\nRank: {i + 1}")
        print("-" * 40)
        print(f"Chunk ID: {c['chunk_id']}")
        if "ce_score" in c:
            print(f"Cross-Encoder Score: {c['ce_score']:.4f}")
        else:
            print(f"Distance: {c['distance']:.4f}")
        print(f"Section: {c['section']}")
        print(f"Page: {c['page']}")
        print(f"Relevance: {rel_label}")
        print(f"Relevance Score: {score}")
        try:
            print(f"Text Preview:\n{c['text'][:800]}")
        except UnicodeEncodeError:
            safe = c["text"][:800].encode("ascii", "replace").decode("ascii")
            print(f"Text Preview:\n{safe}")

    relevance_pct = (total_score / k) * 100
    precision_at_k = (relevant_count / k) * 100

    print(f"\n{'=' * 60}")
    print(f"TOP-{k} SUMMARY {label}")
    print(f"{'=' * 60}")
    print(f"\nRetrieved:        {k}")
    print(f"Relevant:         {relevant_count}")
    print(f"Partially Relevant: {partial_count}")
    print(f"Not Relevant:     {not_relevant_count}")
    print(f"\nRelevance %:      {relevance_pct:.2f}%")
    print(f"Precision@{k}:    {precision_at_k:.2f}%")

    return relevance_pct, precision_at_k


def main():
    print("=" * 60)
    print("Cross-Encoder Reranking Experiment")
    print("=" * 60)
    print(f"\nQuery:\n{QUERY}\n")

    # Load models
    print("Loading BGE embedding model...")
    bge_model = SentenceTransformer(BGE_MODEL, device="cpu")

    print("Loading Cross-Encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    print("Models loaded.\n")

    # Stage 1: Vector retrieval
    print("Stage 1: ChromaDB vector retrieval (Top-10)...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_text = QUERY_PREFIX + QUERY
    query_embedding = bge_model.encode(
        query_text, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"section": {"$ne": "Front matter"}}
    )

    candidates = []
    for i in range(len(results["ids"][0])):
        candidates.append({
            "chunk_id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "section": results["metadatas"][0][i]["section"],
            "page": results["metadatas"][0][i]["page"],
            "text": results["documents"][0][i],
        })

    print(f"Retrieved {len(candidates)} candidates.\n")

    # Show before reranking
    print("=" * 60)
    print("BEFORE RERANKING (ChromaDB cosine distance)")
    print("=" * 60)
    for i, c in enumerate(candidates):
        rel_label, _ = RELEVANCE_LABELS.get(
            c["chunk_id"], ("Not Evaluated", 0.0)
        )
        print(f"  #{i+1} {c['chunk_id']} dist={c['distance']:.4f} "
              f"[{rel_label}] {c['section'][:50]}")

    # Stage 2: Cross-Encoder reranking
    print("\nStage 2: Cross-Encoder reranking...")
    pairs = [(QUERY, c["text"]) for c in candidates]
    ce_scores = reranker.predict(pairs)

    for c, score in zip(candidates, ce_scores):
        c["ce_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)

    print("Done.\n")

    # Show after reranking
    print("=" * 60)
    print("AFTER RERANKING (Cross-Encoder score)")
    print("=" * 60)
    for i, c in enumerate(reranked):
        rel_label, _ = RELEVANCE_LABELS.get(
            c["chunk_id"], ("Not Evaluated", 0.0)
        )
        print(f"  #{i+1} {c['chunk_id']} ce={c['ce_score']:.4f} "
              f"[{rel_label}] {c['section'][:50]}")

    # Evaluate all three K values
    for k in [3, 5, 10]:
        evaluate(reranked, k, label="(RERANKED)")


if __name__ == "__main__":
    main()
