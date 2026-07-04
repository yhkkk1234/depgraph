#!/bin/bash
# ============================================================
# Qwen3-VL-8B LoRA Fine-Tuning with CoT data
# ============================================================

WORKDIR="/root/autodl-tmp"
DATA_DIR="$WORKDIR/cot_format"
MODEL_DIR="$WORKDIR/models"
OUTPUT="$WORKDIR/output_qwen3_diagram"

echo "========================================"
echo " Step 1: Environment"
echo "========================================"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers accelerate peft datasets bitsandbytes pillow qwen-vl-utils -q

echo ""
echo "========================================"
echo " Step 2: Download Qwen3-VL-8B"
echo "========================================"
export MODELSCOPE_CACHE="$MODEL_DIR"
pip install modelscope -q
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen3-VL-8B-Instruct', cache_dir='$MODEL_DIR')
print('Done.')
"
MODEL_PATH=$(find "$MODEL_DIR" -name "config.json" -path "*Qwen3-VL*" | head -1 | xargs dirname)
echo "Model: $MODEL_PATH"

echo ""
echo "========================================"
echo " Step 3: Training"
echo "========================================"

python << 'PYEOF'
import json, os, torch, gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()

MODEL_PATH = os.popen("find /root/autodl-tmp/models -name config.json -path '*Qwen3-VL*' | head -1 | xargs dirname").read().strip()
DATA_DIR = "/root/autodl-tmp/cot_format"
OUTPUT = "/root/autodl-tmp/output_qwen3_diagram"
os.makedirs(OUTPUT, exist_ok=True)

# ---------- Load model ----------
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

print(f"Loading model from {MODEL_PATH}...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.base_model.gradient_checkpointing_enable()
model.train()
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor.image_processor.min_pixels = 65536
processor.image_processor.max_pixels = 262144

# ---------- Load dataset ----------
print("\nLoading CoT dataset...")
with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
    samples = [json.loads(line) for line in f if line.strip()]

perfect = [s for s in samples if s.get("metadata", {}).get("score", {}).get("quality") == "perfect"]
good = [s for s in samples if s.get("metadata", {}).get("score", {}).get("quality") == "good"]
samples = perfect + good
print(f"Total: {len(samples)} (perfect: {len(perfect)}, good: {len(good)})")

class DiagramDataset(Dataset):
    def __init__(self, samples, data_dir):
        self.samples = samples
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        msgs = s["messages"]
        images = s.get("images", [])

        # Build Qwen3 chat format with image references
        chat = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if "<image>" in content and images and role == "user":
                img_path = os.path.join(self.data_dir, images[0])
                img = Image.open(img_path).convert("RGB")
                parts = []
                parts.append({"type": "image", "image": img, "min_pixels": 65536, "max_pixels": 262144})
                parts.append({"type": "text", "text": content.replace("<image>", "").strip()})
                chat.append({"role": role, "content": parts})
            else:
                chat.append({"role": role, "content": content})

        text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(chat, image_patch_size=16)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = DiagramDataset(samples, DATA_DIR)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------- Training ----------
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 5
total_steps = len(dataloader) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"Training: {len(dataset)} samples, {total_steps} steps, {num_epochs} epochs\n")

global_step = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        if isinstance(batch, list):
            batch = batch[0]
        batch = {k: v.to(model.device) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"]

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1
        epoch_loss += loss.item()
        if global_step % 10 == 0:
            torch.cuda.empty_cache()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if global_step % 50 == 0:
            model.save_pretrained(os.path.join(OUTPUT, f"checkpoint-{global_step}"))

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

model.save_pretrained(OUTPUT)
processor.save_pretrained(OUTPUT)
print(f"\nModel saved to {OUTPUT}")
PYEOF

echo "========================================"
echo " 训练完成！"
echo " 模型: $OUTPUT"
echo "========================================"
