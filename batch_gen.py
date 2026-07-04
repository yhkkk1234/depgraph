"""
Batch training data generator.
Scans multiple Python packages, finds high-impact functions,
generates bug scenarios, and collects model responses.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from call_graph import analyze_call_graph, suggest_bugs
from data_gen import (
    BugScenario, generate_training_sample, call_model,
    build_system_prompt, score_response, OUTPUT as TRAINING_OUTPUT,
)
from config import API_KEY, API_BASE, MODEL_NAME
from scan_deps import scan_project
from render_diagram import render_dependency_graph

os.makedirs(TRAINING_OUTPUT, exist_ok=True)


def discover_scenarios(project_name: str, project_path: str, max_scenarios: int = 5) -> list[BugScenario]:
    """Auto-discover bug scenarios from a project's call graph."""
    print(f"  Analyzing call graph for {project_name}...")
    cg = analyze_call_graph(project_path)

    # Build source cache
    source_cache = {}
    root = Path(project_path)
    for fpath in root.rglob("*.py"):
        if "__pycache__" not in str(fpath):
            try:
                source_cache[str(fpath.relative_to(root))] = fpath.read_text(encoding="utf-8")
            except Exception:
                pass

    suggestions = suggest_bugs(cg, source_cache, top_n=20)

    # Generic names to skip (too ambiguous for the model)
    generic_names = {"get", "set", "copy", "send", "write", "read", "close",
                     "open", "run", "call", "exec", "update", "delete", "items",
                     "keys", "values", "pop", "push", "add", "remove", "clear",
                     "append", "extend", "insert", "index", "count", "sort",
                     "reverse", "start", "stop", "wait", "sleep", "print",
                     "decode", "encode", "load", "dump", "save", "parse"}

    scenarios = []

    for sug in suggestions:
        if len(scenarios) >= max_scenarios:
            break

        func_name = sug["function"]
        caller_files = sug["caller_files"]

        # Skip generic names
        if func_name.lower() in generic_names:
            continue

        # Skip functions with too many callers (>15 hard for the model)
        if len(caller_files) > 15:
            continue

        if not sug["bug_ideas"]:
            continue

        def_file = sug["defined_in"][0]["file"] if sug["defined_in"] else ""
        source_code = source_cache.get(def_file, f"# Source not found for {func_name}")

        for idea in sug["bug_ideas"][:1]:
            sid = f"{project_name}_{func_name}_{idea['type']}"
            scenarios.append(BugScenario(
                id=sid,
                project_name=project_name,
                project_path=project_path,
                bug_module=sug["defined_in"][0]["module"] if sug["defined_in"] else func_name,
                bug_func=func_name,
                change_desc=idea["change"],
                source_code=source_code[:5000],
                affected_files=caller_files,
            ))

    print(f"  -> {len(scenarios)} scenarios for {project_name}")
    return scenarios


def batch_collect(projects: list[tuple[str, str]], per_project: int = 3,
                  output_name: str = "batch_dataset") -> str:
    """Discover scenarios across projects and collect training data."""
    all_scenarios = []

    for name, path in projects:
        print(f"\n{'='*50}")
        print(f"Project: {name} ({path})")
        scenarios = discover_scenarios(name, path, max_scenarios=per_project)
        all_scenarios.extend(scenarios)

    print(f"\n{'='*50}")
    print(f"Total scenarios: {len(all_scenarios)} across {len(projects)} projects")
    print(f"Estimated time: {len(all_scenarios) * 50 / 60:.0f} min (at ~50s each)")

    samples = []
    stats = {"total": len(all_scenarios), "perfect": 0, "good": 0,
             "partial": 0, "poor": 0, "failed": 0}

    out_path = os.path.join(TRAINING_OUTPUT, f"{output_name}.jsonl")

    for i, scenario in enumerate(all_scenarios):
        print(f"\n[{i+1}/{len(all_scenarios)}] {scenario.id}")
        sample = generate_training_sample(scenario, retries=2)

        if sample:
            samples.append(sample)
            q = sample["metadata"]["score"]["quality"]
            stats[q] = stats.get(q, 0) + 1
        else:
            stats["failed"] += 1
            # Still try to save partial results
            print("  Failed, continuing...")

        # Save incrementally
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Progress
        elapsed = time.time()
        print(f"  Progress: {len(samples)}/{i+1} collected | "
              f"Perfect: {stats.get('perfect',0)} Good: {stats.get('good',0)}")

    # Final stats
    stats_path = os.path.join(TRAINING_OUTPUT, f"{output_name}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"Dataset: {out_path} ({len(samples)} samples)")
    print(f"Quality: {stats}")
    return out_path


# ── Pre-configured project list ───────────────────────────────

def get_available_projects() -> list[tuple[str, str]]:
    """Find installed Python packages suitable for training data generation."""
    projects = []
    candidates = ["click", "rich", "requests", "httpx", "starlette", "yaml"]

    for name in candidates:
        try:
            mod = __import__(name)
            path = os.path.dirname(mod.__file__)
            py_count = len(list(Path(path).rglob("*.py")))
            if py_count >= 5:
                projects.append((name, path))
                print(f"  {name}: {path} ({py_count} .py files)")
        except Exception:
            pass

    return projects


if __name__ == "__main__":
    if API_KEY == "your-api-key-here":
        print("Set API_KEY in config.py first.")
        sys.exit(1)

    start = time.time()
    print("Auto-discovering projects...")
    projects = get_available_projects()
    print(f"\nFound {len(projects)} projects")

    if not projects:
        print("No suitable projects found.")
        sys.exit(1)

    batch_collect(projects, per_project=3, output_name="auto_batch_1")
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
