import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open(r'C:\Users\ASUS\Desktop\ahmed\Ai-hackathon\embeddings\data\chunks\chunks.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        c = json.loads(line)
        print(f'=== CHUNK {i+1} ===')
        for k, v in c.items():
            val_str = str(v)
            if k == 'text':
                print(f'  {k} ({type(v).__name__}): [{len(val_str)} chars] {val_str[:500]}...' if len(val_str) > 500 else f'  {k} ({type(v).__name__}): [{len(val_str)} chars] {val_str}')
            else:
                print(f'  {k} ({type(v).__name__}): {v}')
        print()
