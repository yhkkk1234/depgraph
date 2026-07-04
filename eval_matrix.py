"""
2x2 Matrix Evaluation:
  Base Model vs Fine-tuned  X  Text Only vs Text+Diagram
"""
import json, os, time, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "/root/autodl-tmp/models"
LORA_PATH = "/root/autodl-tmp/output_diagram_fix"
DATA_DIR = "/root/autodl-tmp/llamafactory_format"
RESULTS = "/root/autodl-tmp/eval_results"
os.makedirs(RESULTS, exist_ok=True)

# Find model path
MODEL_PATH = str(next(Path(MODEL_DIR).rglob("config.json")).parent)
for root, _, files in os.walk(MODEL_DIR):
    if "Qwen2-VL" in root and "config.json" in files:
        MODEL_PATH = root; break

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

def load_processor():
    p = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    p.image_processor.min_pixels = 65536
    p.image_processor.max_pixels = 262144
    return p

def ask(model, processor, prompt: str, img_path: str = None) -> str:
    if img_path:
        img = Image.open(img_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    else:
        msgs = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img] if img_path else None, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)

def score(resp: str, expected: list[str]) -> dict:
    r = resp.lower()
    found = [f for f in expected if f.lower() in r]
    return {"found": len(found), "total": len(expected), "missing": [f for f in expected if f.lower() not in r]}

# ── Build test cases ──
with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
    samples = [json.loads(line) for line in f if line.strip()]

cases = []
for s in samples[:20]:
    meta = s.get("metadata", {})
    affected = meta.get("affected_files", [])
    if not affected or len(affected) < 3:
        continue
    bug_mod = meta.get("bug_module", "?")
    bug_func = meta.get("bug_func", "?")
    change = meta.get("change_desc", "")

    prompt = f"""You are a senior software engineer fixing a bug.

PROJECT: click
MODULE CHANGED: {bug_mod}
FUNCTION: {bug_func}()
CHANGE: {change}

This change breaks all callers. Identify ALL files that call {bug_func}() and would break."""

    img_p = os.path.join(DATA_DIR, s["images"][0])
    if os.path.exists(img_p):
        cases.append({"id": f"{bug_mod}_{bug_func}", "prompt": prompt, "img": img_p, "expected": affected})

print(f"Loaded {len(cases)} hard test cases\n")

# ── Phase 1: Base Model ──
print("=" * 50)
print("PHASE 1: BASE MODEL (no fine-tuning)")
print("=" * 50)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
base_model.eval()
processor = load_processor()

base_text = []; base_img = []
for i, c in enumerate(cases):
    print(f"[{i+1}/{len(cases)}] {c['id']}")
    t = ask(base_model, processor, c["prompt"])
    s_t = score(t, c["expected"])
    base_text.append(s_t)

    t2 = ask(base_model, processor, c["prompt"], c["img"])
    s_t2 = score(t2, c["expected"])
    base_img.append(s_t2)
    print(f"  Text: {s_t['found']}/{s_t['total']}  Diagram: {s_t2['found']}/{s_t2['total']}")
    time.sleep(0.3)

del base_model; torch.cuda.empty_cache()

# ── Phase 2: Fine-tuned Model ──
print("\n" + "=" * 50)
print("PHASE 2: FINE-TUNED MODEL")
print("=" * 50)
ft_model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
ft_model = PeftModel.from_pretrained(ft_model, LORA_PATH)
ft_model.eval()
processor = load_processor()

ft_text = []; ft_img = []
for i, c in enumerate(cases):
    print(f"[{i+1}/{len(cases)}] {c['id']}")
    t = ask(ft_model, processor, c["prompt"])
    s_t = score(t, c["expected"])
    ft_text.append(s_t)

    t2 = ask(ft_model, processor, c["prompt"], c["img"])
    s_t2 = score(t2, c["expected"])
    ft_img.append(s_t2)
    print(f"  Text: {s_t['found']}/{s_t['total']}  Diagram: {s_t2['found']}/{s_t2['total']}")
    time.sleep(0.3)

# ── Matrix Report ──
def stats(lst):
    total = sum(r["found"] for r in lst)
    maxv = sum(r["total"] for r in lst)
    return total, maxv

bt_f, bt_m = stats(base_text)
bi_f, bi_m = stats(base_img)
ft_f, ft_m = stats(ft_text)
fi_f, fi_m = stats(ft_img)

report = f"""
{'='*60}
2x2 MATRIX EVALUATION
{'='*60}
Cases: {len(cases)}

                    Text Only         Text + Diagram
Base Model      {bt_f:3d}/{bt_m:<3d} ({bt_f/bt_m*100:.0f}%)        {bi_f:3d}/{bi_m:<3d} ({bi_f/bi_m*100:.0f}%)
Fine-tuned      {ft_f:3d}/{ft_m:<3d} ({ft_f/ft_m*100:.0f}%)        {fi_f:3d}/{fi_m:<3d} ({fi_f/fi_m*100:.0f}%)

Base Model:       diagram adds +{bi_f-bt_f} files
Fine-tuned Model: diagram adds +{fi_f-ft_f} files

Habit Transfer:    text-only improved by +{ft_f-bt_f} files ({ft_f/ft_m*100 - bt_f/bt_m*100:.0f}pp)
Diagram Synergy:   fine-tuned+diagram vs base+text = +{fi_f-bt_f} files ({fi_f/fi_m*100 - bt_f/bt_m*100:.0f}pp)
{'='*60}
"""
print(report)
with open(os.path.join(RESULTS, "matrix_report.txt"), "w") as f:
    f.write(report)
print(f"Saved to {RESULTS}/matrix_report.txt")
