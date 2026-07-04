"""
Hard evaluation — the prompt does NOT list affected files.
Only the diagram reveals dependency structure.
"""
import json, os, sys, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "/root/autodl-tmp/models"
LORA_PATH = "/root/autodl-tmp/output_diagram_fix"
DATA_DIR = "/root/autodl-tmp/llamafactory_format"
RESULTS = "/root/autodl-tmp/eval_results"
os.makedirs(RESULTS, exist_ok=True)

MODEL_PATH = next(Path(MODEL_DIR).rglob("config.json"))
MODEL_PATH = str(MODEL_PATH.parent)
for root, _, files in os.walk(MODEL_DIR):
    if "Qwen2-VL" in root and "config.json" in files:
        MODEL_PATH = root; break

print(f"Loading model...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.image_processor.min_pixels = 65536
processor.image_processor.max_pixels = 262144

def ask_text(prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)

def ask_diagram(prompt: str, img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)

def score(resp: str, expected: list[str]) -> dict:
    r = resp.lower()
    found = [f for f in expected if f.lower() in r]
    missing = [f for f in expected if f.lower() not in r]
    return {"found": len(found), "total": len(expected), "missing": missing}

# ── Build HARD test cases: no file list in text ──
with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
    all_samples = [json.loads(line) for line in f if line.strip()]

test_cases = []
for s in all_samples[:20]:
    meta = s.get("metadata", {})
    affected = meta.get("affected_files", [])
    if not affected:
        continue
    bug_module = meta.get("bug_module", "?")
    bug_func = meta.get("bug_func", "?")
    change_desc = meta.get("change_desc", "")

    # Build prompt WITHOUT listing affected files
    hard_prompt = f"""You are a senior software engineer fixing a bug.

PROJECT: click
MODULE CHANGED: {bug_module}
FUNCTION: {bug_func}()
CHANGE: {change_desc}

This change breaks all callers that don't handle the new behavior.
Identify ALL files that call {bug_func}() and would break."""

    img_path = os.path.join(DATA_DIR, s["images"][0]) if s.get("images") else None
    if img_path and os.path.exists(img_path):
        test_cases.append({
            "id": f"{bug_module}_{bug_func}",
            "prompt": hard_prompt,
            "img_path": img_path,
            "affected_files": affected,
        })

print(f"\nTesting {len(test_cases)} HARD cases (no file list in prompt)...\n")

c_results = []; e_results = []

for i, tc in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['affected_files'][:3]}...)")
    
    c_resp = ask_text(tc["prompt"])
    c_sc = score(c_resp, tc["affected_files"])
    c_results.append(c_sc)

    e_resp = ask_diagram(tc["prompt"], tc["img_path"])
    e_sc = score(e_resp, tc["affected_files"])
    e_results.append(e_sc)

    print(f"  Text only:    {c_sc['found']}/{c_sc['total']} files")
    print(f"  With diagram: {e_sc['found']}/{e_sc['total']} files")
    time.sleep(0.3)

# ── Report ──
c_total = sum(r["found"] for r in c_results)
c_max = sum(r["total"] for r in c_results)
e_total = sum(r["found"] for r in e_results)
e_max = sum(r["total"] for r in e_results)

report = f"""
{'='*50}
HARD EVALUATION (no file list in text prompt)
{'='*50}
Cases: {len(test_cases)}

CONTROL (text only):
  Found: {c_total}/{c_max} ({c_total/c_max*100:.0f}%)
  Avg:   {c_total/len(c_results):.1f}/{c_max/len(c_results):.1f} per case

EXPERIMENT (text + diagram):
  Found: {e_total}/{e_max} ({e_total/e_max*100:.0f}%)
  Avg:   {e_total/len(e_results):.1f}/{e_max/len(e_results):.1f} per case

DIAGRAM IMPROVEMENT: +{e_total-c_total} more files
"""

print(report)
with open(os.path.join(RESULTS, "hard_eval.txt"), "w") as f:
    f.write(report)
    json.dump({"control": [{"found": r["found"], "total": r["total"]} for r in c_results],
               "experiment": [{"found": r["found"], "total": r["total"]} for r in e_results]},
              open(os.path.join(RESULTS, "hard_results.json"), "w"), indent=2)
print("Saved.")
