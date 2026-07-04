"""
Fine-tuned Model Evaluation: Control vs. Diagram A/B Test
Run on AutoDL. Loads the fine-tuned LoRA model and tests
whether it shows "global perspective" when fixing bugs.
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

# ── Load model ──
MODEL_PATH = next(Path(MODEL_DIR).rglob("config.json"))
MODEL_PATH = str(MODEL_PATH.parent)
if "qwen" not in MODEL_PATH.lower():
    for root, _, files in os.walk(MODEL_DIR):
        if "Qwen2-VL" in root and "config.json" in files:
            MODEL_PATH = root; break

print(f"Model: {MODEL_PATH}")
print(f"LoRA: {LORA_PATH}")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.image_processor.min_pixels = 65536
processor.image_processor.max_pixels = 262144

def ask_text(prompt: str) -> str:
    """Ask model with text only (control group)."""
    msgs = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)

def ask_with_diagram(prompt: str, img_path: str) -> str:
    """Ask model with text + diagram (experiment group)."""
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)

def score_response(resp: str, expected_files: list[str]) -> dict:
    """Check how many expected files are mentioned."""
    found = [f for f in expected_files if f.lower() in resp.lower()]
    return {"found": len(found), "total": len(expected_files), "missing": [f for f in expected_files if f not in found]}

# ── Build test cases from dataset ──
with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
    all_samples = [json.loads(line) for line in f if line.strip()]

test_cases = []
for s in all_samples[:20]:
    meta = s.get("metadata", {})
    affected = meta.get("affected_files", [])
    if not affected:
        continue
    for msg in s["messages"]:
        if msg["role"] == "user":
            content = msg["content"]
            prompt = " ".join(p["text"] for p in content if p["type"] == "text") if isinstance(content, list) else content
            break
    img_path = os.path.join(DATA_DIR, s["images"][0]) if s.get("images") else None
    case_id = f"{meta.get('bug_module','?')}_{meta.get('bug_func','?')}"
    test_cases.append({
        "id": case_id,
        "prompt": prompt,
        "img_path": img_path,
        "affected_files": affected,
    })

print(f"\nTesting {len(test_cases)} cases...\n")

results = {"control": [], "experiment": []}

for i, tc in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] {tc['id']}")

    # Control: text only
    c_resp = ask_text(tc["prompt"])
    c_score = score_response(c_resp, tc["affected_files"])
    results["control"].append({"id": tc["id"], "score": c_score})

    # Experiment: text + diagram
    if tc["img_path"] and os.path.exists(tc["img_path"]):
        e_resp = ask_with_diagram(tc["prompt"], tc["img_path"])
        e_score = score_response(e_resp, tc["affected_files"])
        results["experiment"].append({"id": tc["id"], "score": e_score})

    c_f = c_score["found"]; ct = c_score["total"]
    e_f = e_score.get("found", 0); et = e_score.get("total", ct)
    print(f"  Control: {c_f}/{ct} | Experiment: {e_f}/{et}")

    time.sleep(0.5)  # Throttle

# ── Summary ──
c_total = sum(r["score"]["found"] for r in results["control"])
c_max = sum(r["score"]["total"] for r in results["control"])
e_total = sum(r["score"]["found"] for r in results["experiment"])
e_max = sum(r["score"]["total"] for r in results["experiment"])

report = f"""
========================================
FINE-TUNED MODEL EVALUATION
========================================
Cases tested: {len(test_cases)}

CONTROL (text only):
  Total files found: {c_total}/{c_max} ({c_total/c_max*100:.0f}%)
  Per-case average: {c_total/len(results['control']):.1f}/{c_max/len(results['control']):.1f}

EXPERIMENT (text + diagram):
  Total files found: {e_total}/{e_max} ({e_total/e_max*100:.0f}%)
  Per-case average: {e_total/len(results['experiment']):.1f}/{e_max/len(results['experiment']):.1f}

IMPROVEMENT: +{e_total-c_total} files ({e_total/e_max*100 - c_total/c_max*100:.0f} percentage points)
========================================
"""

print(report)
with open(os.path.join(RESULTS, "eval_report.txt"), "w") as f:
    f.write(report)
    json.dump(results, open(os.path.join(RESULTS, "eval_results.json"), "w"), indent=2)
print(f"Results saved to {RESULTS}")
