"""
Generalized experiment runner for any Python project.
Takes a project path and a bug description, runs A/B test.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import API_KEY, API_BASE, MODEL_NAME
from scan_deps import scan_project
from render_diagram import render_dependency_graph

RESULTS = os.path.join(os.path.dirname(__file__), "results", "generalized")
os.makedirs(RESULTS, exist_ok=True)


def call_api(messages: list, max_tokens: int = 8192) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        return {"error": f"API error {r.status_code}: {r.text[:300]}"}
    d = r.json()
    return {"content": d["choices"][0]["message"].get("content", ""),
            "reasoning": d["choices"][0]["message"].get("reasoning_content", ""),
            "usage": d.get("usage", {}), "model": d.get("model", "")}


def build_prompt(project_name: str, bug_module: str, bug_func: str,
                 change_desc: str, source_code: str,
                 affected_files: list[str], project_path: str) -> str:
    files_list = "\n".join(f"  - {f}" for f in affected_files)
    return f"""PROJECT: {project_name}
Location: {project_path}

BUG REPORT:
  The function `{bug_func}()` in module `{bug_module}` was changed:
  {change_desc}

  This breaks all callers that don't pass the new required argument.

SOURCE CODE OF {bug_module}:
```python
{source_code}
```

FILES THAT IMPORT FROM {bug_module}:
{files_list}

TASK:
  1. Identify ALL files that call {bug_func}() and would break due to this change.
  2. Count the total number of affected files and call sites.
  3. Show the fix for every affected call site.
  4. Explain your impact analysis process.
"""


def run_experiment(project_path: str, project_name: str,
                   bug_module: str, bug_func: str,
                   change_desc: str, source_code: str,
                   affected_files: list[str]):
    prompt = build_prompt(project_name, bug_module, bug_func, change_desc,
                          source_code, affected_files, project_path)

    # Scan and render diagram
    graph = scan_project(project_path)
    
    png_path = os.path.join(RESULTS, f"{project_name}_graph.png")
    raw_b64 = render_dependency_graph(graph, highlight_module=bug_module,
                                       output_path=png_path)
    raw_b64 = render_dependency_graph(graph, highlight_module=bug_module)
    print(f"  Diagram: {png_path} ({len(raw_b64)} chars b64)")

    # Control: text only
    print("\n  [Control] Text only...")
    c_start = time.time()
    c_result = call_api([{"role": "system", "content": "You are a senior software engineer. Identify ALL affected files and fix every call site. Be thorough."},
                          {"role": "user", "content": [{"type": "text", "text": prompt}]}])
    c_elapsed = time.time() - c_start

    # Experiment: text + diagram
    print("  [Experiment] Text + Diagram...")
    e_start = time.time()
    e_result = call_api([{"role": "system", "content": "You are a senior software engineer with access to a system dependency diagram. FIRST analyze the diagram to identify all impacted modules, THEN fix every call site."},
                          {"role": "user", "content": [
                              {"type": "text", "text": "BELOW IS THE PROJECT DEPENDENCY DIAGRAM. The red node is the changed module. Yellow nodes are direct dependents.\n\n" + prompt},
                              {"type": "image_url", "image_url": {"url": raw_b64}}]}])
    e_elapsed = time.time() - e_start

    # Score
    expected = set(affected_files)
    c_mentioned = 0; e_mentioned = 0; c_fp = 0; e_fp = 0

    if "content" in c_result:
        c_text = c_result["content"]
        c_mentioned = sum(1 for f in expected if f in c_text)
        c_fp_files = {"main.py", "__init__.py", "setup.py"}
        c_fp = sum(1 for f in c_fp_files if f in c_text and f not in expected)

    if "content" in e_result:
        e_text = e_result["content"]
        e_mentioned = sum(1 for f in expected if f in e_text)
        e_fp = sum(1 for f in c_fp_files if f in e_text and f not in expected)

    # Report
    report = f"""
{'='*60}
GENERALIZED EXPERIMENT: {project_name}
{'='*60}
Bug: {bug_func}() in {bug_module}
Expected affected files: {len(expected)}
  {chr(10).join(f'  - {f}' for f in sorted(expected))}

SCORE:
  Control    (text only):     {c_mentioned}/{len(expected)} files, {c_fp} false positives
  Experiment (text + diagram): {e_mentioned}/{len(expected)} files, {e_fp} false positives

TOKEN USAGE:
  Control:    {c_result.get('usage', {}).get('total_tokens', '?')}
  Experiment: {e_result.get('usage', {}).get('total_tokens', '?')}

TIME:
  Control:    {c_elapsed:.1f}s
  Experiment: {e_elapsed:.1f}s
"""
    print(report)
    
    # Save results
    for name, data in [("control", c_result), ("experiment", e_result)]:
        p = os.path.join(RESULTS, f"{project_name}_{name}.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(data.get("content", json.dumps(data)))
    
    rp = os.path.join(RESULTS, f"{project_name}_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Full results: {RESULTS}")


if __name__ == "__main__":
    import click as _click
    proj = os.path.dirname(_click.__file__)
    utils_src = open(os.path.join(proj, "utils.py"), encoding="utf-8").read()

    run_experiment(
        project_path=proj,
        project_name="click",
        bug_module="utils",
        bug_func="echo",
        change_desc="Changed from `def echo(message, file=None, ...)` to `def echo(message, file, ...)` — 'file' is now a required positional argument.",
        source_code=utils_src,
        affected_files=[
            "_termui_impl.py", "_winconsole.py", "core.py", "decorators.py",
            "exceptions.py", "globals.py", "shell_completion.py",
            "termui.py", "testing.py", "types.py",
        ],
    )
