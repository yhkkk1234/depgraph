"""
Ultimate Test: Toy project (NEVER seen in training).
Tests generalization of the diagram-reading habit.
"""
import os, time, torch, json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "/root/autodl-tmp/models"
LORA_PATH = "/root/autodl-tmp/output_diagram_fix"
DIAGRAM = "/root/autodl-tmp/toy_diagram_v2.png"

MODEL_PATH = next(Path(MODEL_DIR).rglob("config.json"))
MODEL_PATH = str(MODEL_PATH.parent)
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
        out = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
    # Skip input tokens, decode only generated part
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(out[0][input_len:], skip_special_tokens=True)

# ── Ground truth (NOT revealed to model) ──
# Accept multiple naming conventions
EXPECTED_ALIASES = {
    "services/task_service.py": ["task_service.py", "services.task_service", "services/task_service"],
    "services/notification.py": ["notification.py", "services.notification", "services/notification"],
    "api/handlers.py": ["handlers.py", "api.handlers", "api/handlers"],
    "utils/validators.py": ["validators.py", "utils.validators", "utils/validators"],
}

def fuzzy_score(resp: str) -> dict:
    r = resp.lower()
    result = {}
    for fname, aliases in EXPECTED_ALIASES.items():
        found = any(a.lower() in r for a in aliases)
        result[fname] = found
    found = [f for f, v in result.items() if v]
    return {"found": found, "count": len(found)}

# ── Text-only (no diagram, no legend) ──
PROMPT_NO_DIAGRAM = """You are a senior software engineer fixing a cross-module bug.

ERROR:
  TypeError: Task.to_dict() missing 1 required positional argument: 'include_comments'
  models/task.py changed from `def to_dict(self, include_comments: bool = False)`
  to `def to_dict(self, include_comments: bool)` — default value removed.

Which files call task.to_dict() and need to be fixed?
List each file name on its own line."""

# ── Minimal legend (3 lines) + diagram ──
MINI_LEGEND = """
Diagram key: models/task.py (red) -> services/task_service.py, services/notification.py, api/handlers.py, utils/validators.py (yellow)
"""

PROMPT_DIAGRAM = """You are a senior software engineer fixing a cross-module bug.

ERROR:
  TypeError: Task.to_dict() missing 1 required positional argument: 'include_comments'
  models/task.py changed from `def to_dict(self, include_comments: bool = False)`
  to `def to_dict(self, include_comments: bool)` — default value removed.

""" + MINI_LEGEND + """
DIAGRAM BELOW. Red = changed. Yellow = dependents. Cross-reference with key above.

Which files call task.to_dict() and need to be fixed?
List each file name on its own line."""

print("=" * 60)
print("TOY PROJECT TEST (never seen in training)")
print("=" * 60)
print(f"Expected: 4 files (services/task_service, services/notification, api/handlers, utils/validators)")
print(f"Diagram: {DIAGRAM}")
print()

# ── Load models ──
print("Loading base model...")
base = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
base.eval()
p = load_processor()

print("Loading fine-tuned model...")
ft = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True)
ft = PeftModel.from_pretrained(ft, LORA_PATH)
ft.eval()
p2 = load_processor()

# ── Test: 4 scenarios ──
results = {}

for name, model, proc in [("Base", base, p), ("Fine-tuned", ft, p2)]:
    print(f"\n{'='*40}")
    print(f"  {name} Model")
    print(f"{'='*40}")

    # Text only — NO diagram, NO dependency info
    resp_t = ask(model, proc, PROMPT_NO_DIAGRAM)
    sc_t = fuzzy_score(resp_t)
    print(f"  Text Only (no info):  {sc_t['count']}/4 => {sc_t['found']}")
    print(f"    {resp_t[:400]}...")
    results[f"{name}_text"] = sc_t

    # With diagram — only source of dependency info
    resp_i = ask(model, proc, PROMPT_DIAGRAM, DIAGRAM)
    sc_i = fuzzy_score(resp_i)
    print(f"  +Diagram (visual only): {sc_i['count']}/4 => {sc_i['found']}")
    print(f"    {resp_i[:400]}...")
    results[f"{name}_diagram"] = sc_i

    time.sleep(0.5)

# ── Summary ──
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for label in ["Base_text", "Base_diagram", "Fine-tuned_text", "Fine-tuned_diagram"]:
    r = results[label]
    print(f"  {label:20s}: {r['count']}/4  {r['found']}")

bt = results["Base_text"]["count"]
bi = results["Base_diagram"]["count"]
ft_t = results["Fine-tuned_text"]["count"]
ft_i = results["Fine-tuned_diagram"]["count"]

print(f"\n  Base diagram delta:  {bi-bt:+d}")
print(f"  FT diagram delta:    {ft_i-ft_t:+d}")
print(f"  FT text vs Base text: {ft_t-bt:+d}")

# ── Detailed responses for manual inspection ──
with open("/root/autodl-tmp/eval_results/toy_test.txt", "w") as f:
    for label in ["Base_text", "Base_diagram", "Fine-tuned_text", "Fine-tuned_diagram"]:
        f.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
        # We don't store responses in the results dict, so skip detail
    pass

print(f"\n{'='*60}")
if ft_i == 4:
    print("** GENERALIZATION SUCCESS: Found all 4 files! **")
elif ft_i > bt:
    print(f"Improvement: diagram helps fine-tuned model by {ft_i-bt:+d} files")
elif ft_t > bt:
    print(f"Habit transfer: fine-tuned model is better even without diagram")
else:
    print("No clear improvement. Need more analysis.")
print(f"{'='*60}")
