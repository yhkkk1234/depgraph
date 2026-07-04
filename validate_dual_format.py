"""
Quick validation with MiMo API: does dual format (diagram + text legend)
improve bug-fix accuracy without any fine-tuning?
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from config import API_KEY, API_BASE, MODEL_NAME
import requests
from scan_deps import scan_project
from render_diagram import render_dependency_graph
from legend_gen import generate_full_legend


def call_api(messages: list) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 2048}
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        return f"API error: {r.status_code}"
    d = r.json()
    return d["choices"][0]["message"].get("content", "")


def score(resp: str, expected: list[str]):
    found = [f for f in expected if f.lower() in resp.lower()]
    return len(found), len(expected), [f for f in expected if f.lower() not in resp.lower()]


# Bug scenario: our toy project
PROJ = os.path.join(os.path.dirname(__file__), "test_project")
graph = scan_project(PROJ)
legend = generate_full_legend(graph, "models.task")
img_b64 = render_dependency_graph(graph, highlight_module="models.task")

EXPECTED = [
    "services/task_service.py",
    "services/notification.py",
    "api/handlers.py",
    "utils/validators.py",
]

sys_prompt = "You are a senior software engineer. Identify ALL files affected by a bug. List each file name on its own line. Be thorough."

bug_desc = """ERROR:
  TypeError: Task.to_dict() missing 1 required positional argument: 'include_comments'
  models/task.py: `def to_dict(self, include_comments: bool = False)` changed to
  `def to_dict(self, include_comments: bool)` — default value removed."""

# ── Test 1: Text only ──
print("=== TEST 1: Text Only (no legend, no diagram) ===")
resp1 = call_api([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [{"type": "text", "text": bug_desc + "\n\nWhich files call task.to_dict() and need to be fixed?"}]}
])
f1, t1, m1 = score(resp1, EXPECTED)
print(f"  Found: {f1}/{t1}  Missing: {m1}")
print(f"  Response: {resp1[:300]}...\n")
time.sleep(1)

# ── Test 2: Text legend only ──
print("=== TEST 2: Text Legend Only ===")
resp2 = call_api([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [{"type": "text", "text": bug_desc + "\n\n" + legend + "\n\nWhich files call task.to_dict()?"}]}
])
f2, t2, m2 = score(resp2, EXPECTED)
print(f"  Found: {f2}/{t2}  Missing: {m2}")
print(f"  Response: {resp2[:300]}...\n")
time.sleep(1)

# ── Test 3: Diagram only ──
print("=== TEST 3: Diagram Only (no legend) ===")
resp3 = call_api([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [
        {"type": "text", "text": bug_desc + "\n\nDIAGRAM BELOW. Red = models/task (changed), Yellow = dependents.\nWhich files call task.to_dict()?"},
        {"type": "image_url", "image_url": {"url": img_b64}},
    ]}
])
f3, t3, m3 = score(resp3, EXPECTED)
print(f"  Found: {f3}/{t3}  Missing: {m3}")
print(f"  Response: {resp3[:300]}...\n")
time.sleep(1)

# ── Test 4: Diagram + Legend (DUAL) ──
print("=== TEST 4: Diagram + Text Legend (DUAL) ===")
resp4 = call_api([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [
        {"type": "text", "text": bug_desc + "\n\n" + legend + "\n\nDIAGRAM BELOW. Cross-reference with legend above.\nWhich files call task.to_dict()?"},
        {"type": "image_url", "image_url": {"url": img_b64}},
    ]}
])
f4, t4, m4 = score(resp4, EXPECTED)
print(f"  Found: {f4}/{t4}  Missing: {m4}")
print(f"  Response: {resp4[:300]}...")

# ── Summary ──
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"  Text only:        {f1}/{t1}")
print(f"  Legend only:      {f2}/{t2}")
print(f"  Diagram only:     {f3}/{t3}")
print(f"  Diagram+Legend:   {f4}/{t4}")
