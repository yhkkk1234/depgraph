"""
Test: Does the model spontaneously call depgraph when facing a cross-module bug?
Uses MiMo's OpenAI-compatible function calling.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from config import API_KEY, API_BASE, MODEL_NAME
import requests

DEPGRAPH_TOOL = {
    "type": "function",
    "function": {
        "name": "depgraph",
        "description": "Generate a module dependency graph for a project. Returns which files import from the target module, showing the impact surface of any change.",
        "parameters": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project name or path"},
                "module": {"type": "string", "description": "The changed module to analyze (e.g. 'models.task')"},
            },
            "required": ["project", "module"],
        },
    },
}

def call_with_tools(prompt: str):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a senior software engineer. You have access to tools. Use them when analyzing cross-module bugs."},
            {"role": "user", "content": prompt},
        ],
        "tools": [DEPGRAPH_TOOL],
        "tool_choice": "auto",
        "max_tokens": 2048,
    }
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        print(f"Error: {r.status_code} {r.text[:300]}")
        return None
    return r.json()

# ── Test 1: Cross-module bug (should trigger tool call) ──
print("=== Test 1: Cross-module bug ===")
result = call_with_tools("""ERROR: TypeError: Task.to_dict() missing 1 required positional argument: 'include_comments'
The Task.to_dict() in models/task.py had its default value removed.

Which files are affected by this change? Be specific.""")

if result:
    msg = result["choices"][0]["message"]
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        print(f"  SPONTANEOUS TOOL CALL: {tc['function']['name']}({tc['function']['arguments']})")
        print("  ** Model asked for depgraph! **")
    else:
        print(f"  NO tool call. Response: {msg.get('content','')[:200]}")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ── Test 2: Simulate tool response → see if model uses it ──
print("\n=== Test 2: Tool → Response loop ===")

# Simulate: model calls depgraph, we return data, model answers
from scan_deps import scan_project
PROJ = os.path.join(os.path.dirname(__file__), "test_project")
graph = scan_project(PROJ)
dep_data = """Files directly dependent on models.task:
  - services/task_service.py
  - services/notification.py
  - api/handlers.py
  - utils/validators.py
  - storage/database.py"""

payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": "You have access to a depgraph tool. Use it for cross-module bugs."},
        {"role": "user", "content": "ERROR: Task.to_dict() signature changed in models/task.py. Which files are affected? List each one."},
        # Simulate tool call + response
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "depgraph", "arguments": '{"project":"toy","module":"models.task"}'}
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": dep_data},
    ],
    "max_tokens": 512,
}
r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
if r.status_code == 200:
    resp = r.json()["choices"][0]["message"]["content"]
    print(f"  Model's answer after tool: {resp[:500]}...")
else:
    print(f"  Error: {r.status_code}")

print("\nDone.")
