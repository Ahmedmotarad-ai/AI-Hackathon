import json
import sys
from pathlib import Path
from collections import Counter


EVAL_FILE = Path("data/evaluation/eval_dataset.json")
CHUNKS_FILE = Path("data/chunks/chunks.jsonl")

VALID_LABELS = {0, 1, 2}
MIN_QUERIES = 15


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            ids.append(record["chunk_id"])
    return ids


def validate_dataset():
    errors = []
    warnings = []

    # Load data
    print("=" * 60)
    print("Evaluation Dataset Validation")
    print("=" * 60)

    if not EVAL_FILE.exists():
        print(f"\nFAIL: File not found: {EVAL_FILE}")
        sys.exit(1)

    dataset = load_json(EVAL_FILE)
    print(f"\nLoaded: {EVAL_FILE}")

    # Load actual chunk IDs
    actual_chunk_ids = load_chunk_ids(CHUNKS_FILE)
    print(f"Actual chunk count: {len(actual_chunk_ids)}")
    actual_chunk_set = set(actual_chunk_ids)

    # ------------------------------------------------------------------
    # 1. Structure validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("1. Structure Validation")
    print(f"{'=' * 60}")

    required_keys = [
        "dataset_version", "document_scope", "chunk_count",
        "relevance_scale", "queries"
    ]
    for key in required_keys:
        if key not in dataset:
            errors.append(f"Missing top-level key: '{key}'")

    queries = dataset.get("queries", [])
    print(f"Queries in dataset: {len(queries)}")

    if len(queries) < MIN_QUERIES:
        errors.append(
            f"Expected at least {MIN_QUERIES} queries, found {len(queries)}"
        )

    declared_chunks = dataset.get("chunk_count", 0)
    if declared_chunks != len(actual_chunk_ids):
        warnings.append(
            f"Declared chunk_count={declared_chunks} but actual={len(actual_chunk_ids)}"
        )

    # ------------------------------------------------------------------
    # 2. Query validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("2. Query Validation")
    print(f"{'=' * 60}")

    query_ids = []
    query_texts = []
    categories = Counter()
    difficulties = Counter()

    for q in queries:
        qid = q.get("query_id", "")
        query_ids.append(qid)
        query_texts.append(q.get("query", ""))
        categories[q.get("category", "unknown")] += 1
        difficulties[q.get("difficulty", "unknown")] += 1

    # Unique query IDs
    dup_ids = [
        qid for qid, count in Counter(query_ids).items() if count > 1
    ]
    if dup_ids:
        errors.append(f"Duplicate query IDs: {dup_ids}")
    else:
        print("  [PASS] All query IDs are unique")

    # No duplicate query texts
    dup_texts = [
        text for text, count in Counter(query_texts).items() if count > 1
    ]
    if dup_texts:
        errors.append(f"Duplicate query texts: {dup_texts}")
    else:
        print("  [PASS] No duplicate query texts")

    # All queries have required fields
    for q in queries:
        for field in ["query_id", "query", "category", "difficulty", "relevance"]:
            if field not in q:
                errors.append(
                    f"Query {q.get('query_id', '?')}: missing field '{field}'"
                )
    print("  [PASS] All queries have required fields")

    print(f"\n  Category distribution:")
    for cat, count in categories.most_common():
        print(f"    {cat}: {count}")

    print(f"\n  Difficulty distribution:")
    for diff, count in difficulties.most_common():
        print(f"    {diff}: {count}")

    # ------------------------------------------------------------------
    # 3. Chunk ID validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("3. Chunk ID Validation")
    print(f"{'=' * 60}")

    all_referenced_ids = set()
    for q in queries:
        qid = q.get("query_id", "?")
        relevance = q.get("relevance", {})
        relevance_ids = set(relevance.keys())
        all_referenced_ids.update(relevance_ids)

        # Check no unknown chunk IDs
        unknown = relevance_ids - actual_chunk_set
        if unknown:
            errors.append(
                f"Query {qid}: unknown chunk IDs: {unknown}"
            )

        # Check all chunks labeled
        missing = actual_chunk_set - relevance_ids
        if missing:
            errors.append(
                f"Query {qid}: missing labels for chunks: {missing}"
            )

    if not errors or not any("unknown chunk" in e for e in errors):
        print("  [PASS] All referenced chunk IDs exist")

    # Check completeness for each query
    complete_count = 0
    for q in queries:
        qid = q.get("query_id", "?")
        relevance = q.get("relevance", {})
        if set(relevance.keys()) == actual_chunk_set:
            complete_count += 1
        else:
            errors.append(
                f"Query {qid}: does not label all {len(actual_chunk_ids)} chunks"
            )

    print(f"  Queries with complete labels: {complete_count}/{len(queries)}")

    # ------------------------------------------------------------------
    # 4. Label validation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("4. Label Validation")
    print(f"{'=' * 60}")

    label_counter = Counter()
    invalid_count = 0

    for q in queries:
        qid = q.get("query_id", "?")
        relevance = q.get("relevance", {})
        for chunk_id, label in relevance.items():
            if label not in VALID_LABELS:
                errors.append(
                    f"Query {qid}, chunk {chunk_id}: "
                    f"invalid label {label} (expected 0, 1, or 2)"
                )
                invalid_count += 1
            else:
                label_counter[label] += 1

    if invalid_count == 0:
        print("  [PASS] All labels are valid (0, 1, or 2)")

    total_labels = sum(label_counter.values())
    print(f"\n  Label distribution across all queries:")
    for label in sorted(label_counter.keys()):
        count = label_counter[label]
        pct = (count / total_labels) * 100 if total_labels > 0 else 0
        names = {0: "Not Relevant", 1: "Partially Relevant", 2: "Relevant"}
        print(f"    {label} ({names[label]}): {count} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Relevance distribution per query
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("5. Per-Query Relevance Distribution")
    print(f"{'=' * 60}")

    queries_with_no_relevant = []
    for q in queries:
        qid = q.get("query_id", "?")
        relevance = q.get("relevance", {})
        counts = Counter(relevance.values())
        has_relevant = counts.get(2, 0) > 0
        print(
            f"  {qid}: R={counts.get(2, 0)}, "
            f"PR={counts.get(1, 0)}, NR={counts.get(0, 0)}"
            f"{'  *** NO RELEVANT CHUNK ***' if not has_relevant else ''}"
        )
        if not has_relevant:
            queries_with_no_relevant.append(qid)

    if queries_with_no_relevant:
        errors.append(
            f"Queries with no Relevant (2) chunks: {queries_with_no_relevant}"
        )
    else:
        print("\n  [PASS] All queries have at least one Relevant (2) chunk")

    # ------------------------------------------------------------------
    # 6. Category coverage
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("6. Section/Topic Coverage")
    print(f"{'=' * 60}")

    chunk_ids_labeled_relevant = Counter()
    for q in queries:
        for chunk_id, label in q.get("relevance", {}).items():
            if label == 2:
                chunk_ids_labeled_relevant[chunk_id] += 1

    never_relevant = actual_chunk_set - set(chunk_ids_labeled_relevant.keys())
    always_irrelevant = {
        cid for cid in actual_chunk_set
        if cid not in chunk_ids_labeled_relevant
    }

    if never_relevant:
        warnings.append(
            f"Chunks never labeled Relevant (2) in any query: "
            f"{sorted(never_relevant)}"
        )
        print(f"\n  Chunks never labeled Relevant: {len(never_relevant)}")
        for cid in sorted(never_relevant):
            print(f"    {cid}")
    else:
        print("\n  [PASS] Every chunk is labeled Relevant (2) in at least one query")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Queries: {len(queries)}")
    print(f"  Chunks per query: {len(actual_chunk_ids)}")
    print(f"  Total relevance judgments: {len(queries) * len(actual_chunk_ids)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    [ERROR] {e}")

    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    [WARN] {w}")

    if errors:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    else:
        print("\nVALIDATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    validate_dataset()
