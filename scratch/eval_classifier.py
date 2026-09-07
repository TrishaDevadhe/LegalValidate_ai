import os, sys, dotenv, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents

dotenv.load_dotenv()
data = json.load(open('evals/test_set.json'))
cats = {}
total = len(data)
correct = 0

for d in data:
    cat = d.get('category', 'CONTRACT')
    if cat not in cats:
        cats[cat] = {'total': 0, 'correct': 0}
    cats[cat]['total'] += 1
    
    res = agents.classify_legal(d['text'])
    is_corr = (res.is_legal == d['expected_is_legal'])
    if is_corr:
        correct += 1
        cats[cat]['correct'] += 1
    else:
        reason_clean = res.reason.encode('ascii', 'ignore').decode() if res.reason else 'No reason'
        print(f"MISMATCH [{cat}] {d['name']}: expected={d['expected_is_legal']}, got={res.is_legal} ({reason_clean})")

print(f"\n==================================================")
print(f"OVERALL CLASSIFIER ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"==================================================")
for c, s in cats.items():
    acc = (s['correct'] / s['total'] * 100) if s['total'] > 0 else 0
    print(f"Category {c:<12}: {s['correct']}/{s['total']} ({acc:.1f}%)")
