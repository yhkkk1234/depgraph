#!/bin/bash
# ============================================================
# Qwen3-VL-8B Quick Test: Can it read diagram labels?
# No training — pure base model evaluation.
# ============================================================

WORKDIR="/root/autodl-tmp"
MODEL_DIR="$WORKDIR/models"

echo "========================================"
echo " Step 1: pip"
echo "========================================"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers accelerate bitsandbytes pillow modelscope qwen-vl-utils -q

echo ""
echo "========================================"
echo " Step 2: Download Qwen3-VL-8B"
echo "========================================"
export MODELSCOPE_CACHE="$MODEL_DIR"
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen3-VL-8B-Instruct', cache_dir='$MODEL_DIR')
print('Done.')
"
MODEL_PATH=$(find "$MODEL_DIR" -name "config.json" -path "*Qwen3-VL*" | head -1 | xargs dirname)
echo "Model: $MODEL_PATH"

echo ""
echo "========================================"
echo " Step 3: Toy project test (pure diagram, NO text key)"
echo "========================================"

python << 'PYEOF'
import torch, os, json, sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

MODEL_PATH = os.popen("find /root/autodl-tmp/models -name config.json -path '*Qwen3-VL*' | head -1 | xargs dirname").read().strip()
DIAGRAM = "/root/autodl-tmp/toy_diagram_v2.png"
EXPECTED = [
    "services/task_service.py",
    "services/notification.py",
    "api/handlers.py",
    "utils/validators.py",
]

print(f"Model: {MODEL_PATH}")
print(f"Diagram: {DIAGRAM}")

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ── Test 1: Text only (no diagram, no legend) ──
msgs1 = [{"role": "user", "content": [
    {"type": "text", "text": """You are a senior software engineer.

ERROR: Task.to_dict() missing 1 required positional argument: 'include_comments'
models/task.py: def to_dict(self, include_comments: bool = False) was changed to
def to_dict(self, include_comments: bool) — default value removed.

Which files call task.to_dict() and need to be fixed? List each on its own line."""}
]}]
text1 = processor.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True)
images1, _ = process_vision_info(msgs1, image_patch_size=16)
inputs1 = processor(text=[text1], images=images1, return_tensors="pt").to(model.device)
with torch.no_grad():
    out1 = model.generate(**inputs1, max_new_tokens=512)
resp1 = processor.decode(out1[0][inputs1["input_ids"].shape[1]:], skip_special_tokens=True)

found1 = [f for f in EXPECTED if f.lower() in resp1.lower()]
print(f"\n=== Test 1: Text Only ===")
print(f"  Found: {len(found1)}/4  => {found1}")
print(f"  Response: {resp1[:400]}...")

# ── Test 2a: List ALL yellow nodes (pure OCR test) ──
img = Image.open(DIAGRAM).convert("RGB")
msgs2a = [{"role": "user", "content": [
    {"type": "image", "image": img, "min_pixels": 256*32*32, "max_pixels": 1280*32*32},
    {"type": "text", "text": """THE DIAGRAM BELOW is a project dependency graph.
Red node = changed module. Yellow nodes = direct dependents.
Gray nodes = unrelated modules.

List EVERY yellow node name you can read from the diagram, one per line.
Do not guess — only list what you can actually see."""}
]}]
text2a = processor.apply_chat_template(msgs2a, tokenize=False, add_generation_prompt=True)
images2a, _ = process_vision_info(msgs2a, image_patch_size=16)
inputs2a = processor(text=[text2a], images=images2a, return_tensors="pt", do_resize=False).to(model.device)
with torch.no_grad():
    out2a = model.generate(**inputs2a, max_new_tokens=512)
resp2a = processor.decode(out2a[0][inputs2a["input_ids"].shape[1]:], skip_special_tokens=True)

YELLOW_EXPECTED = ["services/task_service.py", "services/notification.py", "api/handlers.py",
                    "storage/database.py", "utils/validators.py"]
found_ocr = [f for f in YELLOW_EXPECTED if f.lower() in resp2a.lower()]
print(f"\n=== Test 2a: Pure OCR — list ALL yellow nodes ===")
print(f"  Expected: 5 yellow nodes")
print(f"  Found: {len(found_ocr)}/5  => {found_ocr}")
print(f"  Response: {resp2a[:600]}...")
msgs2 = [{"role": "user", "content": [
    {"type": "image", "image": img, "min_pixels": 256*32*32, "max_pixels": 1280*32*32},
    {"type": "text", "text": """You are a senior software engineer.

ERROR: Task.to_dict() missing 1 required positional argument: 'include_comments'
models/task.py changed — default value removed.

THE DIAGRAM BELOW is the project dependency graph.
Red = models.task (changed). Yellow = its direct dependents.
Module names are labeled on each node.

Which files call task.to_dict() and need to be fixed? List each on its own line."""}
]}]
text2 = processor.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
images2, _ = process_vision_info(msgs2, image_patch_size=16)
inputs2 = processor(text=[text2], images=images2, return_tensors="pt").to(model.device)
with torch.no_grad():
    out2 = model.generate(**inputs2, max_new_tokens=1024)
resp2 = processor.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)

found2 = [f for f in EXPECTED if f.lower() in resp2.lower()]
print(f"\n=== Test 2: Diagram Only (no key) ===")
print(f"  Found: {len(found2)}/4  => {found2}")
print(f"  Response: {resp2[:400]}...")

# ── Summary ──
print(f"\n{'='*50}")
print(f"Qwen2-VL-7B (before):  Text=0/4, Diagram=0/4")
print(f"Qwen3-VL-8B (now):     Text={len(found1)}/4, OCR={len(found_ocr)}/5 yellow, Bug={len(found2)}/4")
if found_ocr:
    print(f"  OCR yellow nodes: {found_ocr}")
    missing_ocr = [f for f in YELLOW_EXPECTED if f not in found_ocr]
    if missing_ocr:
        print(f"  Could not read: {missing_ocr} — diagram layout issue")
if len(found2) == 4:
    print("** OCR BOTTLENECK BROKEN **")
print(f"{'='*50}")
PYEOF
