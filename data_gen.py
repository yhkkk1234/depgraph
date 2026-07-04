"""
Training Data Generation Pipeline.

Produces instruction-tuning data for teaching models to internalize
"diagram → impact analysis → fix" workflow.

Output format (JSONL):
{
  "id": "bug_001",
  "project": "click",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "bug_module": "utils", "bug_func": "echo",
    "affected_files": [...], "files_found": 10, "quality": "perfect"
  }
}
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(__file__))
from config import API_KEY, API_BASE, MODEL_NAME
from scan_deps import scan_project
from render_diagram import render_dependency_graph

OUTPUT = os.path.join(os.path.dirname(__file__), "training_data")
os.makedirs(OUTPUT, exist_ok=True)


@dataclass
class BugScenario:
    """A controlled bug for training data generation."""
    id: str
    project_name: str
    project_path: str
    bug_module: str
    bug_func: str
    change_desc: str
    source_code: str
    affected_files: list[str]
    expected_analysis: str = ""
    quality: str = "pending"


def call_model(messages: list, max_tokens: int = 8192) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.2,
               "max_tokens": max_tokens}
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=300)
    if r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:200]}", "content": ""}
    d = r.json()
    msg = d["choices"][0]["message"]
    return {"content": msg.get("content", ""),
            "reasoning": msg.get("reasoning_content", ""),
            "usage": d.get("usage", {}),
            "model": d.get("model", "")}


def build_system_prompt() -> str:
    return (
        "You are an expert software engineer trained in system-level impact analysis. "
        "For every code change request, you MUST follow this workflow:\n"
        "1. STUDY the dependency diagram — identify the changed module (red) and its dependents (yellow).\n"
        "2. For EACH dependent module, determine whether it calls the changed function.\n"
        "3. List ALL affected files with specific line numbers or call site descriptions.\n"
        "4. Provide the exact fix for each call site.\n"
        "5. Summarize with: total files affected, total call sites fixed.\n"
        "NEVER fix only the file mentioned in the error — use the diagram to find ALL affected files."
    )


def build_user_prompt(scenario: BugScenario) -> tuple[str, str]:
    """Returns (text_prompt, image_b64)."""
    files_list = "\n".join(f"  - {f}" for f in sorted(scenario.affected_files))
    text = f"""PROJECT: {scenario.project_name}

BUG: `{scenario.bug_func}()` in `{scenario.bug_module}` was changed:
  {scenario.change_desc}

SOURCE OF {scenario.bug_module}:
```python
{scenario.source_code}
```

FILES THAT IMPORT FROM THIS MODULE:
{files_list}

STUDY THE DEPENDENCY DIAGRAM BELOW, then identify ALL files that call {scenario.bug_func}() and fix every one.
"""

    graph = scan_project(scenario.project_path)
    img_b64 = render_dependency_graph(graph, highlight_module=scenario.bug_module)

    return text, img_b64


def score_response(response_text: str, expected_files: list[str]) -> dict:
    """Score a model response against expected results."""
    text_lower = response_text.lower()
    found = [f for f in expected_files if f.lower() in text_lower]
    missing = [f for f in expected_files if f.lower() not in text_lower]

    # Check if response shows structured analysis (has step-by-step reasoning)
    has_structured = any(kw in text_lower for kw in
                         ["step 1", "step 2", "impact analysis", "dependency diagram",
                          "affected module", "dependent", "first"])

    # Check if fix code is present
    has_fixes = "```python" in response_text or "```diff" in response_text

    quality = "poor"
    if len(found) == len(expected_files):
        quality = "perfect"
    elif len(found) >= len(expected_files) * 0.75:
        quality = "good"
    elif len(found) >= len(expected_files) * 0.5:
        quality = "partial"

    return {
        "files_found": len(found),
        "files_expected": len(expected_files),
        "files_missing": missing,
        "has_structured_analysis": has_structured,
        "has_fix_code": has_fixes,
        "quality": quality,
    }


def generate_training_sample(scenario: BugScenario, retries: int = 2) -> Optional[dict]:
    """Generate one training sample. Retry if quality is poor."""
    text_prompt, img_b64 = build_user_prompt(scenario)

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ]},
    ]

    for attempt in range(retries):
        print(f"  [{scenario.id}] Attempt {attempt + 1}/{retries}...")
        result = call_model(messages)
        
        if "error" in result and not result.get("content"):
            print(f"    API error: {result['error'][:100]}")
            continue

        score = score_response(result["content"], scenario.affected_files)
        print(f"    Score: {score['files_found']}/{score['files_expected']} files, "
              f"quality={score['quality']}")

        if score["quality"] in ("perfect", "good"):
            # Build the training record with the assistant response
            training_messages = [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": img_b64}},
                ]},
                {"role": "assistant", "content": result["content"]},
            ]

            return {
                "id": scenario.id,
                "project": scenario.project_name,
                "messages": training_messages,
                "metadata": {
                    "bug_module": scenario.bug_module,
                    "bug_func": scenario.bug_func,
                    "change_desc": scenario.change_desc,
                    "affected_files": scenario.affected_files,
                    "score": score,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", ""),
                },
            }

        print(f"    Retrying...")

    return None


def collect_dataset(scenarios: list[BugScenario], output_name: str = "dataset") -> str:
    """Run all scenarios and collect training data."""
    samples = []
    stats = {"total": len(scenarios), "perfect": 0, "good": 0, "failed": 0}

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {scenario.id}: {scenario.project_name}/{scenario.bug_func}")
        
        sample = generate_training_sample(scenario)
        if sample:
            samples.append(sample)
            q = sample["metadata"]["score"]["quality"]
            stats[q] = stats.get(q, 0) + 1
        else:
            stats["failed"] += 1

        # Save incrementally
        out_path = os.path.join(OUTPUT, f"{output_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save stats
    stats_path = os.path.join(OUTPUT, f"{output_name}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset: {out_path}")
    print(f"Total: {stats['total']} | Perfect: {stats.get('perfect',0)} | "
          f"Good: {stats.get('good',0)} | Failed: {stats['failed']}")
    return out_path


# ── Bug Scenario Factory ──────────────────────────────────────

def create_click_scenarios() -> list[BugScenario]:
    """Generate bug scenarios for the click project."""
    import click as _c
    proj = os.path.dirname(_c.__file__)

    utils_src = open(os.path.join(proj, "utils.py"), encoding="utf-8").read()
    core_src = open(os.path.join(proj, "core.py"), encoding="utf-8").read()
    types_src = open(os.path.join(proj, "types.py"), encoding="utf-8").read()

    return [
        BugScenario(
            id="click_echo_file_required",
            project_name="click",
            project_path=proj,
            bug_module="utils",
            bug_func="echo",
            change_desc="Changed `def echo(message, file=None, ...)` to `def echo(message, file, ...)` — 'file' is now required positional.",
            source_code=utils_src,
            affected_files=[
                "_termui_impl.py", "_winconsole.py", "core.py", "decorators.py",
                "exceptions.py", "globals.py", "shell_completion.py",
                "termui.py", "testing.py", "types.py",
            ],
        ),
        BugScenario(
            id="click_echo_nl_required",
            project_name="click",
            project_path=proj,
            bug_module="utils",
            bug_func="echo",
            change_desc="Changed `def echo(message, file=None, nl=True, ...)` to `def echo(message, file=None, nl, ...)` — 'nl' is now required positional.",
            source_code=utils_src,
            affected_files=[
                "_termui_impl.py", "_winconsole.py", "core.py", "decorators.py",
                "exceptions.py", "globals.py", "shell_completion.py",
                "termui.py", "testing.py", "types.py",
            ],
        ),
        BugScenario(
            id="click_format_filename_required",
            project_name="click",
            project_path=proj,
            bug_module="utils",
            bug_func="format_filename",
            change_desc="Added required parameter `max_length: int` to `format_filename()`.",
            source_code=utils_src,
            affected_files=["exceptions.py", "types.py"],
        ),
    ]


if __name__ == "__main__":
    if API_KEY == "your-api-key-here":
        print("Set API_KEY in config.py first.")
        sys.exit(1)

    scenarios = create_click_scenarios()
    print(f"Generated {len(scenarios)} bug scenarios\n")
    collect_dataset(scenarios, output_name="click_echo")
