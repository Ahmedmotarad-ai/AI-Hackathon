# Section-Aware Chunking

This implementation takes the parsed output from Person 2 and produces `chunks.jsonl`.

## Input
`02_parsed_documents.json`

## Output
`chunks.jsonl`

## Strategy
- Section detection is based on the actual extracted text, not only the upstream `section` field.
- Noisy/truncated page section labels are corrected when the text contains a reliable section heading.
- A page is kept intact whenever possible.
- Consecutive pages belonging to the same section are grouped up to about 3200 characters.
- A page containing multiple main sections is split at the section boundary.
- Long source blocks are split only at a newline/sentence boundary.
- Section hierarchy is stored in `section_path`.
- Page traceability is preserved with `page`, `page_start`, and `page_end`.
- No source content is summarized or rewritten.

## Why no fixed overlap?
The source parser already gives page-level boundaries, and the chunker avoids splitting pages/recommendation blocks whenever possible. For this guideline, preserving complete clinical context is prioritized over duplicating text through aggressive overlap. Overlap can be added later during retrieval experiments if evaluation shows a benefit.

## Run

```bash
python chunker.py
```
