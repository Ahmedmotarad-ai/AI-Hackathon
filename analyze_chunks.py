import json
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

docs = defaultdict(list)
with open(r'C:\Users\ASUS\Desktop\ahmed\Ai-hackathon\embeddings\data\chunks\chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        c = json.loads(line)
        docs[c['document']].append(c)

topics = [
    ('HFmrEF', r'HFmrEF|hf.mref|mildly reduced ejection fraction'),
    ('HFpEF', r'HFpEF|hf.pef|preserved ejection fraction'),
    ('SGLT2 inhibitors', r'SGLT2|sglt2|sodium-glucose co-transporter 2'),
    ('dapagliflozin', r'dapagliflozin'),
    ('empagliflozin', r'empagliflozin'),
    ('acute heart failure', r'acute heart failure|acute HF'),
    ('beta-blockers', r'beta.blocker|beta blocker'),
    ('ACE inhibitors', r'ACE inhibitor|ACEi'),
    ('ARNI', r'ARNI|angiotensin receptor.*neprilysin'),
    ('MRA', r'MRA |mineralocorticoid|aldosterone receptor'),
    ('diagnosis', r'diagnos'),
    ('symptoms', r'symptom|breathless|oedema|fatigue'),
    ('palliative care', r'palliative|end.of.life|advance care'),
    ('cardiac rehabilitation', r'cardiac rehab|rehabilitation'),
]

for topic, pattern in topics:
    print(f'\n===== TOPIC: {topic} =====')
    found = []
    for doc in sorted(docs.keys()):
        for c in docs[doc]:
            if re.search(pattern, c['text'], re.IGNORECASE):
                found.append(c)
    if not found:
        print('  (no matches)')
        continue

    by_doc = defaultdict(list)
    for c in found:
        by_doc[c['document']].append(c)

    for doc in sorted(by_doc.keys()):
        chunks = by_doc[doc]
        picks = [chunks[0]]
        if len(chunks) > 2:
            picks.append(chunks[len(chunks)//2])
        elif len(chunks) > 1:
            picks.append(chunks[1])
        for c in picks:
            preview = c['text'][:200].replace('\n', ' ')
            cid = c['chunk_id']
            sec = c['section']
            pg = c['page']
            print(f'  [{doc}]')
            print(f'    chunk_id={cid}  section={sec}  page={pg}')
            print(f'    text(200): {preview}')
            print()
    print(f'  Total matching chunks across all docs: {len(found)}')
