"""
Parallel high-throughput training data collector.
Uses ThreadPoolExecutor for concurrent API calls.
"""
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from call_graph import analyze_call_graph, suggest_bugs
from data_gen import (
    BugScenario, generate_training_sample, OUTPUT as TRAINING_OUTPUT,
)
from config import API_KEY, API_BASE, MODEL_NAME

os.makedirs(TRAINING_OUTPUT, exist_ok=True)
_write_lock = threading.Lock()

GENERIC_NAMES = {
    "get", "set", "copy", "send", "write", "read", "close",
    "open", "run", "call", "exec", "update", "delete", "items",
    "keys", "values", "pop", "push", "add", "remove", "clear",
    "append", "extend", "insert", "index", "count", "sort",
    "reverse", "start", "stop", "wait", "sleep", "print",
    "decode", "encode", "load", "dump", "save", "parse",
    "join", "split", "find", "replace", "format", "strip",
}


def discover_all_scenarios(project_name: str, project_path: str,
                           max_scenarios: int = 10) -> list[BugScenario]:
    """Discover scenarios from a project's call graph."""
    cg = analyze_call_graph(project_path)

    source_cache = {}
    root = Path(project_path)
    for fpath in root.rglob("*.py"):
        if "__pycache__" not in str(fpath):
            try:
                source_cache[str(fpath.relative_to(root))] = fpath.read_text(encoding="utf-8")
            except Exception:
                pass

    suggestions = suggest_bugs(cg, source_cache, top_n=30)
    scenarios = []

    for sug in suggestions:
        if len(scenarios) >= max_scenarios:
            break

        func_name = sug["function"]
        caller_files = sug["caller_files"]

        if func_name.lower() in GENERIC_NAMES:
            continue
        if len(caller_files) < 3 or len(caller_files) > 15:
            continue
        if not sug["bug_ideas"]:
            continue

        def_file = sug["defined_in"][0]["file"] if sug["defined_in"] else ""
        source_code = source_cache.get(def_file, f"# N/A")

        for idea in sug["bug_ideas"][:2]:  # Try 2 bug types per function
            if len(scenarios) >= max_scenarios:
                break
            sid = f"{project_name}_{func_name}_{idea['type']}"
            scenarios.append(BugScenario(
                id=sid,
                project_name=project_name,
                project_path=project_path,
                bug_module=sug["defined_in"][0]["module"] if sug["defined_in"] else project_name,
                bug_func=func_name,
                change_desc=idea["change"],
                source_code=source_code[:5000],
                affected_files=caller_files,
            ))

    return scenarios


def process_scenario(scenario: BugScenario, idx: int, total: int) -> dict | None:
    """Process one scenario. Returns training sample or None."""
    print(f"  [{idx}/{total}] {scenario.id} ({scenario.project_name})")
    sample = generate_training_sample(scenario, retries=2)
    if sample:
        q = sample["metadata"]["score"]["quality"]
        print(f"    -> {q} ({sample['metadata']['score']['files_found']}/{sample['metadata']['score']['files_expected']} files)")
    else:
        print(f"    -> FAILED")
    return sample


def find_all_projects() -> list[tuple[str, str]]:
    """Find all suitable Python packages in site-packages."""
    import site
    projects = []
    site_dirs = site.getsitepackages()
    
    for site_dir in site_dirs:
        if not os.path.isdir(site_dir):
            continue
        for entry in sorted(os.listdir(site_dir)):
            full = os.path.join(site_dir, entry)
            if not os.path.isdir(full):
                continue
            py_count = len(list(Path(full).rglob("*.py")))
            # Filter: 8-200 .py files, no huge frameworks
            if 8 <= py_count <= 200:
                if entry not in ("pip", "setuptools", "wheel", "pkg_resources",
                                 "numpy", "pandas", "matplotlib", "scipy",
                                 "torch", "tensorflow", "opencode"):
                    projects.append((entry, full))
    
    return projects


def parallel_collect(projects: list[tuple[str, str]], per_project: int = 8,
                     output_name: str = "parallel_dataset",
                     workers: int = 6) -> str:
    """Discover scenarios and collect in parallel."""
    print(f"\nPhase 1: Discovering scenarios across {len(projects)} projects...")
    all_scenarios = []

    for name, path in projects:
        try:
            scenarios = discover_all_scenarios(name, path, max_scenarios=per_project)
            all_scenarios.extend(scenarios)
            print(f"  {name}: {len(scenarios)} scenarios")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    print(f"\nPhase 2: Collecting {len(all_scenarios)} scenarios "
          f"with {workers} parallel workers...")
    est_time = len(all_scenarios) * 50 / workers / 60
    print(f"Estimated time: {est_time:.0f} min\n")

    samples = []
    total = len(all_scenarios)
    out_path = os.path.join(TRAINING_OUTPUT, f"{output_name}.jsonl")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, scenario in enumerate(all_scenarios):
            f = executor.submit(process_scenario, scenario, i + 1, total)
            futures[f] = i

        for f in as_completed(futures):
            sample = f.result()
            if sample:
                with _write_lock:
                    samples.append(sample)
                    with open(out_path, "w", encoding="utf-8") as fout:
                        for s in samples:
                            fout.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    stats = {}
    for s in samples:
        q = s["metadata"]["score"]["quality"]
        stats[q] = stats.get(q, 0) + 1
    stats["failed"] = total - len(samples)

    print(f"\n{'='*60}")
    print(f"DONE: {out_path}")
    print(f"Collected: {len(samples)}/{total}")
    print(f"Quality: {stats}")
    return out_path


if __name__ == "__main__":
    if API_KEY == "your-api-key-here":
        print("Set API_KEY first.")
        sys.exit(1)

    start = time.time()
    print("Scanning site-packages for projects...")
    all_projects = find_all_projects()
    print(f"Found {len(all_projects)} candidate projects")

    # Filter for ones with good success rates or known quality
    priority = ["click", "requests", "httpx", "starlette", "rich", "yaml",
                "flask", "fastapi", "pydantic", "jinja2", "markupsafe",
                "certifi", "charset_normalizer", "idna", "urllib3"]
    projects = [(n, p) for n, p in all_projects if n in priority]
    # Add any extra
    seen = {n for n, _ in projects}
    for n, p in all_projects:
        if n not in seen:
            projects.append((n, p))
            seen.add(n)
        if len(projects) >= 30:
            break

    print(f"Selected {len(projects)} projects for collection\n")
    parallel_collect(projects, per_project=5, output_name="parallel_run",
                     workers=8)
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")
