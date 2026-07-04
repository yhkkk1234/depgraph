"""
Multi-modal AI Code Fix Experiment: Control vs. Visual Diagram

Tests the hypothesis: providing a system dependency diagram to a multi-modal
LLM improves its "global perspective" when fixing cross-module bugs.

Design:
  Control Group:   code + error traceback only
  Experiment Group: code + error traceback + dependency diagram (image)
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from config import (
    API_KEY, API_BASE, MODEL_NAME, TEST_PROJECT,
    OUTPUT_DIR, EXPECTED_FIXES, TOTAL_FIX_SITES,
)
from scan_deps import scan_project, graph_to_mermaid
from render_diagram import render_dependency_graph

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _read_file(rel_path: str) -> str:
    return Path(TEST_PROJECT, rel_path).read_text(encoding="utf-8")


def _call_api(messages: list, temperature: float = 0.2) -> dict:
    """Call OpenAI-compatible multi-modal API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
    }
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )
    if resp.status_code != 200:
        return {"error": f"API error {resp.status_code}: {resp.text[:500]}"}
    data = resp.json()
    msg = data["choices"][0]["message"]
    return {
        "content": msg.get("content", ""),
        "reasoning": msg.get("reasoning_content", ""),
        "usage": data.get("usage", {}),
        "model": data.get("model", ""),
    }


def _count_fixes(response_text: str) -> dict:
    """Score the response by checking which affected files are mentioned
    and whether the model identified them as needing fixes.

    Returns: {by_file: {mentioned, fixed}, total_mentioned, total_fixed, false_positives}
    """
    found = {}
    total_mentioned = 0
    total_fixed = 0
    false_positives = []

    # Files the model might incorrectly flag
    legit_files = set(EXPECTED_FIXES.keys())
    all_possible = [
        "services/task_service.py", "services/notification.py",
        "api/handlers.py", "utils/validators.py",
        "models/user.py", "main.py", "storage/database.py",
    ]

    # Check each legitimate file
    for file_rel in legit_files:
        mentioned = file_rel in response_text
        # Check if model suggests a fix for this file
        has_fix = mentioned and any(
            kw in response_text.split(file_rel)[-1][:500] if file_rel in response_text else ""
            for kw in ["include_comments", "to_dict(False", "to_dict(include"]
        ) if mentioned else False
        found[file_rel] = {"mentioned": mentioned, "has_fix": has_fix}
        if mentioned:
            total_mentioned += 1
        if has_fix:
            total_fixed += 1

    # Check for false positives
    for f in all_possible:
        if f not in legit_files and f in response_text:
            # Check if it was flagged as needing fix (not just mentioned in context)
            false_positives.append(f)

    return {
        "by_file": found,
        "total_mentioned": total_mentioned,
        "total_fixed": total_fixed,
        "total_expected": len(EXPECTED_FIXES),
        "false_positives": false_positives,
    }


def _fmt_time(val) -> str:
    """Format elapsed time, handling None/missing values."""
    if isinstance(val, (int, float)):
        return f"{val:.1f}s"
    return "N/A"


def build_error_info() -> str:
    """Build the error description for the experiment."""
    task_code = _read_file("models/task.py")
    return f"""PROJECT STRUCTURE:
  models/task.py       - Task model with to_dict() method
  models/user.py       - User model
  services/task_service.py  - Task CRUD operations
  services/notification.py  - Notification service
  storage/database.py  - In-memory database
  api/handlers.py      - API request handlers
  utils/validators.py  - Input validators and formatters
  main.py              - Entry point

ERROR:
  Traceback (most recent call last):
    File "main.py", line 17, in main
      r1 = handlers.handle_create_task(...)
    File "api/handlers.py", line 26, in handle_create_task
      return {{"success": True, "task": task.to_dict()}}
  TypeError: Task.to_dict() missing 1 required positional argument: 'include_comments'

CAUSE:
  The method signature in models/task.py was changed from:
    def to_dict(self, include_comments: bool = False) -> dict:
  to:
    def to_dict(self, include_comments: bool) -> dict:
  The default value was removed, making include_comments a required parameter.
  All call sites that don't pass include_comments now break.

SOURCE OF MODELS/TASK.PY:
```python
{task_code}
```

TASK:
  1. Identify ALL files that call task.to_dict() and need to be fixed.
  2. State clearly how many files and how many call sites you found.
  3. Show the diff or modified code for each affected location.
  4. Explain your reasoning for the fix strategy.
"""


def run_control_group(error_info: str) -> dict:
    """Control: code + error only, no diagram."""
    print("\n" + "=" * 60)
    print("CONTROL GROUP: Code + Error Only (No Diagram)")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior software engineer. Given a bug report and source code, "
                "identify ALL affected files and fix every call site. Be thorough — "
                "a partial fix will leave hidden bugs."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": error_info},
            ],
        },
    ]

    start = time.time()
    result = _call_api(messages)
    elapsed = time.time() - start

    if "error" in result:
        return result

    fix_stats = _count_fixes(result["content"])
    return {
        **result,
        "group": "control",
        "elapsed_sec": elapsed,
        "fix_stats": fix_stats,
    }


def run_experiment_group(error_info: str, image_b64: str) -> dict:
    """Experiment: code + error + dependency diagram (image)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT GROUP: Code + Error + Dependency Diagram")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior software engineer with access to a system dependency "
                "diagram. For every bug fix, FIRST analyze the diagram to identify all "
                "impacted modules, THEN fix every affected call site. The diagram is your "
                "'global canvas' — use it to avoid tunnel vision on a single file."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "BELOW IS A SYSTEM DEPENDENCY DIAGRAM showing module relationships. "
                        "Modules pointing to 'models_task' are consumers that may be affected.\n\n"
                        + error_info
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_b64},
                },
            ],
        },
    ]

    start = time.time()
    result = _call_api(messages)
    elapsed = time.time() - start

    if "error" in result:
        return result

    fix_stats = _count_fixes(result["content"])
    return {
        **result,
        "group": "experiment",
        "elapsed_sec": elapsed,
        "fix_stats": fix_stats,
    }


def generate_report(control: dict, experiment: dict) -> str:
    """Generate a comparison report."""
    c_fix = control.get("fix_stats", {})
    e_fix = experiment.get("fix_stats", {})

    report = []
    report.append("=" * 60)
    report.append("EXPERIMENT REPORT: Visual Diagram vs. Text-Only Bug Fixing")
    report.append("=" * 60)
    report.append(f"Timestamp: {datetime.now().isoformat()}")
    report.append(f"Model: {MODEL_NAME}")
    report.append(f"Total fix sites needed: {TOTAL_FIX_SITES}")
    report.append("")

    report.append("-" * 60)
    report.append("SCORE COMPARISON")
    report.append("-" * 60)
    c_total = c_fix.get("total_mentioned", 0) if "error" not in control else "ERR"
    e_total = e_fix.get("total_mentioned", 0) if "error" not in experiment else "ERR"
    c_fixed = c_fix.get("total_fixed", 0) if "error" not in control else "ERR"
    e_fixed = e_fix.get("total_fixed", 0) if "error" not in experiment else "ERR"
    c_fp = len(c_fix.get("false_positives", [])) if isinstance(c_fix, dict) else "?"
    e_fp = len(e_fix.get("false_positives", [])) if isinstance(e_fix, dict) else "?"
    report.append(f"  Control Group    : {c_total}/{TOTAL_FIX_SITES} files identified, {c_fixed} fixes shown, {c_fp} false positives")
    report.append(f"  Experiment Group : {e_total}/{TOTAL_FIX_SITES} files identified, {e_fixed} fixes shown, {e_fp} false positives")

    if isinstance(c_total, int) and isinstance(e_total, int):
        diff = e_total - c_total
        if diff > 0:
            report.append(f"  Improvement        : +{diff} more files identified")
        elif diff == 0:
            report.append("  Improvement        : Same files identified, check false positives")
        else:
            report.append(f"  Regression          : {diff}")

    report.append("")
    report.append("-" * 60)
    report.append("PER-FILE BREAKDOWN (identified/fix shown)")
    report.append("-" * 60)
    for file_rel in EXPECTED_FIXES:
        c_info = c_fix.get("by_file", {}).get(file_rel, {}) if isinstance(c_fix, dict) else {}
        e_info = e_fix.get("by_file", {}).get(file_rel, {}) if isinstance(e_fix, dict) else {}
        c_m = "Y" if c_info.get("mentioned") else "N"
        c_f = "Y" if c_info.get("has_fix") else "N"
        e_m = "Y" if e_info.get("mentioned") else "N"
        e_f = "Y" if e_info.get("has_fix") else "N"
        report.append(f"  {file_rel:35s}  Control: {c_m}/{c_f}  Experiment: {e_m}/{e_f}")

    report.append("")
    report.append("-" * 60)
    report.append("TOKEN USAGE")
    report.append("-" * 60)
    for name, result in [("Control", control), ("Experiment", experiment)]:
        usage = result.get("usage", {})
        report.append(f"  {name}: prompt={usage.get('prompt_tokens', '?')}, "
                      f"completion={usage.get('completion_tokens', '?')}, "
                      f"total={usage.get('total_tokens', '?')}")

    report.append("")
    report.append("-" * 60)
    report.append("ELAPSED TIME")
    report.append("-" * 60)
    report.append(f"  Control:    {_fmt_time(control.get('elapsed_sec'))}")
    report.append(f"  Experiment: {_fmt_time(experiment.get('elapsed_sec'))}")

    return "\n".join(report)


def main():
    print("Multi-modal Code Fix Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Project: {TEST_PROJECT}")

    if API_KEY == "your-api-key-here":
        print("\nERROR: Set your API key in config.py or EXPERIMENT_API_KEY env var.")
        sys.exit(1)

    # Step 1: Scan project, build dependency graph
    print("\n[1/4] Scanning project dependencies...")
    graph = scan_project(TEST_PROJECT)
    graph_path = os.path.join(OUTPUT_DIR, "dependency_graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"  Found {len(graph['modules'])} modules")
    print(f"  Saved graph to {graph_path}")

    # Step 2: Generate dependency diagram PNG
    print("\n[2/4] Generating dependency diagram...")
    mermaid_code = graph_to_mermaid(graph, highlight="models.task")
    mermaid_path = os.path.join(OUTPUT_DIR, "dependency_graph.mmd")
    with open(mermaid_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    print(f"  Mermaid code saved to {mermaid_path}")

    png_path = os.path.join(OUTPUT_DIR, "dependency_graph.png")
    render_dependency_graph(
        graph, highlight_module="models.task", output_path=png_path
    )
    # Get raw base64 (no data: prefix) for MiMo API compatibility
    raw_b64 = render_dependency_graph(graph, highlight_module="models.task")
    print(f"  PNG saved to {png_path}")
    print(f"  Base64 length: {len(raw_b64)} chars")

    # Step 3: Build error info (shared between groups)
    print("\n[3/4] Building error scenario...")
    error_info = build_error_info()
    error_path = os.path.join(OUTPUT_DIR, "error_scenario.txt")
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(error_info)
    print(f"  Error scenario saved to {error_path}")
    print(f"  Bug: Task.to_dict() signature changed, {TOTAL_FIX_SITES} call sites affected")

    # Step 4: Run both groups
    print(f"\n[4/4] Running experiment ({TOTAL_FIX_SITES} fix sites across {len(EXPECTED_FIXES)} files)...")

    control_result = run_control_group(error_info)
    c_out = os.path.join(OUTPUT_DIR, "control_response.txt")
    with open(c_out, "w", encoding="utf-8") as f:
        f.write(control_result.get("content", json.dumps(control_result)))
    print(f"  Control response saved to {c_out}")

    experiment_result = run_experiment_group(error_info, raw_b64)
    e_out = os.path.join(OUTPUT_DIR, "experiment_response.txt")
    with open(e_out, "w", encoding="utf-8") as f:
        f.write(experiment_result.get("content", json.dumps(experiment_result)))
    print(f"  Experiment response saved to {e_out}")

    # Step 5: Generate report
    print("\n" + "=" * 60)
    report = generate_report(control_result, experiment_result)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
