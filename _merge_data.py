import json, os

td = os.path.join(os.path.dirname(__file__), "training_data")
all_samples = []
for fname in os.listdir(td):
    if fname.endswith(".jsonl"):
        with open(os.path.join(td, fname), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))

seen = set()
unique = []
for s in all_samples:
    if s["id"] not in seen:
        seen.add(s["id"])
        unique.append(s)

quality_order = {"perfect": 0, "good": 1, "partial": 2, "poor": 3}
unique.sort(key=lambda s: quality_order.get(s["metadata"]["score"]["quality"], 99))

merged_path = os.path.join(td, "merged_dataset.jsonl")
with open(merged_path, "w", encoding="utf-8") as f:
    for s in unique:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

projects = {}
for s in unique:
    p = s["project"]
    projects[p] = projects.get(p, 0) + 1

perfect = sum(1 for s in unique if s["metadata"]["score"]["quality"] == "perfect")
good = sum(1 for s in unique if s["metadata"]["score"]["quality"] == "good")

print(f"Merged: {merged_path}")
print(f"Total: {len(unique)} samples")
print(f"Quality: {perfect} perfect, {good} good")
print(f"Size: {os.path.getsize(merged_path):,} bytes")
print()
print("By project:")
for p, c in sorted(projects.items()):
    print(f"  {p:15s}: {c}")
print()
print("By quality:")
for s in unique:
    q = s["metadata"]["score"]["quality"]
    f_count = s["metadata"]["score"]["files_found"]
    f_total = s["metadata"]["score"]["files_expected"]
    print(f"  [{q:7s}] {s['id']:50s} {f_count}/{f_total} files")
