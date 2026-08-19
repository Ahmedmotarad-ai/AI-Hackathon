import sys
sys.path.insert(0, ".")
from safety_gate import classify_query
from confidence_bridge import build_evidence_input
from refusal_handler import get_refusal_decision
from confidence_policy import ConfidenceResult

print("=== safety_gate ===")
sg_tests = [
    ("dose of furosemide", "high_risk", False),
    ("manage HF reduced EF", "medical_clinical", True),
    ("overdose pills", "high_risk", False),
    ("treatment for diabetes", "out_of_scope", False),
    ("", "out_of_scope", False),
    ("tell me about heart failure", "medical_clinical", True),
    ("normal EF range", "medical_clinical", True),
    ("how many mg digoxin", "high_risk", False),
    ("recommend antibiotic", "out_of_scope", False),
    ("HFpEF guidelines", "medical_clinical", True),
]
p = 0
for q, ec, es in sg_tests:
    r = classify_query(q)
    ok = r.classification == ec and r.safe == es
    p += ok
    if not ok:
        print(f"  FAIL: '{q}' got ({r.classification},{r.safe}) exp ({ec},{es})")
print(f"  {p}/{len(sg_tests)} passed")

print("\n=== confidence_bridge ===")
chunk_ids = ["nice_hf_2018_chunk_0021", "nice_hf_2018_chunk_0023", "nice_hf_2018_chunk_0025"]
ce_scores = [6.3, 5.97, 5.77]
full_chunks = {
    "nice_hf_2018_chunk_0021": {"guideline_family": "NICE_HF", "superseded_by": "", "parent_guideline": ""},
    "nice_hf_2018_chunk_0023": {"guideline_family": "NICE_HF", "superseded_by": "", "parent_guideline": ""},
    "nice_hf_2018_chunk_0025": {"guideline_family": "NICE_HF", "superseded_by": "", "parent_guideline": ""},
}
ev = build_evidence_input(chunk_ids, ce_scores, full_chunks, top_k=3)
assert ev.supporting_chunks == 3, f"expected 3 supporting, got {ev.supporting_chunks}"
assert ev.conflicting_sources == False, f"expected no conflict"
assert 0 < ev.retrieval_quality <= 1.0
assert 0 <= ev.coverage <= 1.0
assert 0 <= ev.score_agreement <= 1.0
assert 0 <= ev.source_consistency <= 1.0
print(f"  quality={ev.retrieval_quality} coverage={ev.coverage} agreement={ev.score_agreement} consistency={ev.source_consistency} supporting={ev.supporting_chunks} conflicting={ev.conflicting_sources}")
print("  PASS")

print("\n=== refusal_handler ===")
rd1 = get_refusal_decision(ConfidenceResult(0.9, "high", "answer", True, []))
assert rd1.action == "answer" and rd1.proceed_to_llm == True
print(f"  high => action={rd1.action} proceed={rd1.proceed_to_llm}: PASS")

rd2 = get_refusal_decision(ConfidenceResult(0.55, "medium", "caution", True, []))
assert rd2.action == "caution" and rd2.proceed_to_llm == True
print(f"  medium => action={rd2.action} proceed={rd2.proceed_to_llm}: PASS")

rd3 = get_refusal_decision(ConfidenceResult(0.3, "low", "refuse", False, []))
assert rd3.action == "refuse" and rd3.proceed_to_llm == False
print(f"  low => action={rd3.action} proceed={rd3.proceed_to_llm}: PASS")

print("\n=== end-to-end: safety => bridge => confidence => refusal ===")
q = "What are the treatment options for heart failure with reduced ejection fraction?"
sg = classify_query(q)
assert sg.safe == True
ev = build_evidence_input(chunk_ids, ce_scores, full_chunks, top_k=3)
from confidence_policy import calculate_confidence
cr = calculate_confidence(ev)
rd = get_refusal_decision(cr)
print(f"  query: '{q[:50]}...'")
print(f"  safety: {sg.classification} safe={sg.safe}")
print(f"  confidence: {cr.level} score={cr.score:.4f} decision={cr.decision}")
print(f"  refusal: {rd.action} proceed_to_llm={rd.proceed_to_llm}")
assert rd.proceed_to_llm == True
print("  PASS")

print("\n=== ALL TESTS PASSED ===")
