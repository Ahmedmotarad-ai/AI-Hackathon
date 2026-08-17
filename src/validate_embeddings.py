import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("data/embeddings/embedded_chunks.jsonl")

EXPECTED_DIMENSION = 384

REQUIRED_FIELDS = [
    "chunk_id",
    "text",
    "document",
    "section",
    "section_path",
    "page",
    "page_start",
    "page_end",
    "embedding",
]


# ============================================================
# Load Embedded Chunks
# ============================================================

def load_embedded_chunks(file_path):

    records = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                )

    return records


# ============================================================
# Validation
# ============================================================

def validate_records(records):

    errors = []

    if not records:
        errors.append("No records found.")

        return errors

    # --------------------------------------------------------
    # Check required fields
    # --------------------------------------------------------

    for index, record in enumerate(records, start=1):

        for field in REQUIRED_FIELDS:

            if field not in record:

                errors.append(
                    f"Record {index}: missing field '{field}'"
                )

    # --------------------------------------------------------
    # Check embeddings
    # --------------------------------------------------------

    for index, record in enumerate(records, start=1):

        embedding = record.get("embedding")

        if embedding is None:
            continue

        if not isinstance(embedding, list):

            errors.append(
                f"Record {index}: embedding is not a list"
            )

            continue

        if len(embedding) != EXPECTED_DIMENSION:

            errors.append(
                f"Record {index}: "
                f"embedding dimension is {len(embedding)}, "
                f"expected {EXPECTED_DIMENSION}"
            )

        if not all(
            isinstance(value, (int, float))
            for value in embedding
        ):

            errors.append(
                f"Record {index}: embedding contains "
                f"non-numeric values"
            )

    # --------------------------------------------------------
    # Check duplicate chunk IDs
    # --------------------------------------------------------

    chunk_ids = [
        record.get("chunk_id")
        for record in records
        if record.get("chunk_id") is not None
    ]

    duplicate_ids = {
        chunk_id
        for chunk_id in chunk_ids
        if chunk_ids.count(chunk_id) > 1
    }

    if duplicate_ids:

        errors.append(
            f"Duplicate chunk IDs found: "
            f"{len(duplicate_ids)}"
        )

    return errors


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Embedding Validation")
    print("=" * 60)

    # Check file exists

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    print(f"\nInput file: {INPUT_FILE}")

    # Load

    records = load_embedded_chunks(
        INPUT_FILE
    )

    print(f"Records found: {len(records)}")

    # Validate

    print("\nRunning validation...")

    errors = validate_records(records)

    # Results

    print("\n" + "=" * 60)

    if errors:

        print("VALIDATION FAILED")
        print("=" * 60)

        for error in errors:

            print(f"❌ {error}")

        raise SystemExit(1)

    else:

        print("VALIDATION PASSED")
        print("=" * 60)

        print("✅ All required metadata fields are present.")
        print("✅ All embeddings are numeric.")
        print("✅ All embeddings have 384 dimensions.")
        print("✅ No duplicate chunk IDs found.")
        print("✅ All embedded chunks are valid.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()