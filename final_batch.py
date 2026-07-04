"""Final high-efficiency batch: only proven projects + best bug type."""
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from call_graph import analyze_call_graph, suggest_bugs
from data_gen import BugScenario, generate_training_sample, OUTPUT as TDIR
from config import API_KEY

os.makedirs(TDIR, exist_ok=True)
_lock = threading.Lock()
GENERIC = {"get","set","copy","send","write","read","close","open","run","call",
           "exec","update","delete","items","keys","values","pop","push","add",
           "remove","clear","append","extend","insert","index","count","sort",
           "reverse","start","stop","wait","sleep","print","decode","encode",
           "load","dump","save","parse","join","split","find","replace","format",
           "strip","app","receive","compile","finalize","process","connect","execut"}


def discover(name, path, max_n=20):
    cg = analyze_call_graph(path)
    cache = {}
    for fp in Path(path).rglob("*.py"):
        if "__pycache__" not in str(fp):
            try: cache[str(fp.relative_to(path))] = fp.read_text(encoding="utf-8")
            except: pass
    suggestions = suggest_bugs(cg, cache, top_n=50)
    scenarios = []
    for sug in suggestions:
        if len(scenarios) >= max_n: break
        fn = sug["function"]
        cf = sug["caller_files"]
        if fn.lower() in GENERIC: continue
        if len(cf) < 3 or len(cf) > 15: continue
        if not sug["bug_ideas"]: continue
        # Only param_required type (highest success rate)
        param_ideas = [i for i in sug["bug_ideas"] if i["type"] == "param_required"]
        if not param_ideas: continue
        idea = param_ideas[0]
        df = sug["defined_in"][0]["file"] if sug["defined_in"] else ""
        src = cache.get(df, "# N/A")
        sid = f"{name}_{fn}_p"
        scenarios.append(BugScenario(
            id=sid, project_name=name, project_path=path,
            bug_module=sug["defined_in"][0]["module"] if sug["defined_in"] else name,
            bug_func=fn, change_desc=idea["change"],
            source_code=src[:5000], affected_files=cf))
    return scenarios


def process(sc, idx, total):
    print(f"  [{idx}/{total}] {sc.id}")
    sample = generate_training_sample(sc, retries=2)
    if sample:
        q = sample["metadata"]["score"]["quality"]
        print(f"    -> {q} ({sample['metadata']['score']['files_found']}/{sample['metadata']['score']['files_expected']})")
    else:
        print(f"    -> FAILED")
    return sample


def get_high_yield_projects():
    import site
    # Only projects that showed good results + their paths
    known_good = {}
    for site_dir in site.getsitepackages():
        if not os.path.isdir(site_dir): continue
        for entry in sorted(os.listdir(site_dir)):
            full = os.path.join(site_dir, entry)
            if not os.path.isdir(full): continue
            py_count = len(list(Path(full).rglob("*.py")))
            if 10 <= py_count <= 200:
                if entry not in ("pip","setuptools","wheel","numpy","pandas",
                                 "matplotlib","scipy","torch","tensorflow"):
                    known_good[entry] = full
    return known_good


if __name__ == "__main__":
    if API_KEY == "your-api-key-here":
        print("Set API_KEY first."); sys.exit(1)

    start = time.time()
    all_projects = get_high_yield_projects()
    print(f"Found {len(all_projects)} candidate projects\n")

    # Phase 1: Discover
    all_scenarios = []
    for name, path in sorted(all_projects.items()):
        try:
            scs = discover(name, path, max_n=8)
            all_scenarios.extend(scs)
            print(f"  {name}: {len(scs)} scenarios")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    print(f"\nPhase 2: Collecting {len(all_scenarios)} scenarios (8 workers)...")
    est = len(all_scenarios) * 45 / 8 / 60
    print(f"Est time: {est:.0f} min\n")

    samples = []
    total = len(all_scenarios)
    out_path = os.path.join(TDIR, "final_batch.jsonl")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process, sc, i+1, total): i for i, sc in enumerate(all_scenarios)}
        for f in as_completed(futures):
            s = f.result()
            if s:
                with _lock:
                    samples.append(s)
                    with open(out_path, "w", encoding="utf-8") as fout:
                        for ss in samples:
                            fout.write(json.dumps(ss, ensure_ascii=False) + "\n")

    stats = {}
    for s in samples:
        q = s["metadata"]["score"]["quality"]
        stats[q] = stats.get(q, 0) + 1
    stats["failed"] = total - len(samples)

    print(f"\nDONE: {out_path}")
    print(f"Collected: {len(samples)}/{total}")
    print(f"Quality: {stats}")
    print(f"Time: {(time.time()-start)/60:.1f} min")
