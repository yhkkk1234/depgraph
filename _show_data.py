import json, os

path = os.path.join(os.path.dirname(__file__), "training_data", "click_echo.jsonl")
with open(path, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

print(f"Dataset: {path}\n")
for s in samples:
    m = s["metadata"]
    sc = m["score"]
    print(f"  {s['id']}")
    print(f"    Quality : {sc['quality']}")
    print(f"    Files   : {sc['files_found']}/{sc['files_expected']}")
    print(f"    Missing : {sc.get('files_missing', [])}")
    print(f"    Tokens  : {m['usage'].get('total_tokens', '?')}")
    print(f"    Size    : {len(json.dumps(s)):,} chars")
    print()
