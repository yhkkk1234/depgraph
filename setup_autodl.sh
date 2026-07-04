#!/bin/bash
# ============================================================
# 纯 PyTorch LoRA 训练 — 不依赖 LLaMA-Factory Trainer
# ============================================================
set -e

WORKDIR="/root/autodl-tmp"
DATA_DIR="$WORKDIR/cot_format"
MODEL_DIR="$WORKDIR/models"
OUTPUT="$WORKDIR/output_diagram_fix"

echo "========================================"
echo " Step 1: pip 国内源 + 安装依赖"
echo "========================================"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers peft accelerate datasets bitsandbytes pillow modelscope qwen-vl-utils -q

echo ""
echo "========================================"
echo " Step 2: 下载模型"
echo "========================================"
export MODELSCOPE_CACHE="$MODEL_DIR"
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-VL-7B-Instruct', cache_dir='$MODEL_DIR')
"
MODEL_PATH=$(find "$MODEL_DIR" -name "config.json" -path "*Qwen2-VL*" | head -1 | xargs dirname)
echo "Model: $MODEL_PATH"

echo ""
echo "========================================"
echo " Step 3: 加载数据集"
echo "========================================"

python -c "
import json
data = []
with open('$DATA_DIR/dataset.jsonl') as f:
    for line in f:
        data.append(json.loads(line.strip()))
print(f'Loaded {len(data)} samples')
"

echo "========================================"
echo " Step 4: 启动训练"
echo "========================================"

python << 'PYEOF'
import json, os, torch, gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()

MODEL_DIR = "/root/autodl-tmp/models"
DATA_DIR = "/root/autodl-tmp/llamafactory_format"
OUTPUT = "/root/autodl-tmp/output_diagram_fix"
os.makedirs(OUTPUT, exist_ok=True)

# ---------- load model ----------
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_PATH = next(Path(MODEL_DIR).rglob("config.json"))
MODEL_PATH = str(MODEL_PATH.parent)
# if nested, find real path
for root, dirs, files in os.walk(MODEL_DIR):
    for f in files:
        if f == "config.json" and "Qwen2-VL" in root:
            MODEL_PATH = root
            break

print(f"Loading model from {MODEL_PATH}...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen2VLForConditionalGeneration.from_pretrained(
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

# ---------- load dataset ----------
print("\nLoading dataset...")
with open(os.path.join(DATA_DIR, "dataset.jsonl")) as f:
    samples = [json.loads(line) for line in f if line.strip()]

# Only use perfect-quality samples
perfect = [s for s in samples if s.get("metadata", {}).get("score", {}).get("quality") == "perfect"]
print(f"Total: {len(samples)}, Perfect: {len(perfect)}, Using perfect only")

class DiagramDataset(Dataset):
    def __init__(self, samples, data_dir, processor):
        self.samples = samples
        self.data_dir = data_dir
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        msgs = s["messages"]
        images = s.get("images", [])

        # Convert to Qwen2-VL chat format
        chat = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            # Replace <image> with actual image objects
            if "<image>" in content and images and role == "user":
                parts = []
                img_path = os.path.join(self.data_dir, images[0])
                img = Image.open(img_path).convert("RGB")
                parts.append({"type": "image", "image": img})
                parts.append({"type": "text", "text": content.replace("<image>", "").strip()})
                chat.append({"role": role, "content": parts})
            else:
                chat.append({"role": role, "content": content})

        # Use processor's chat template to generate proper text + vision tokens
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

        # Get images from user messages
        imgs = []
        for msg in chat:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image":
                        imgs.append(part["image"])

        inputs = self.processor(
            text=[text], images=imgs if imgs else None,
            return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = DiagramDataset(perfect, DATA_DIR, processor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------- training ----------
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 5
total_steps = len(dataloader) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"\nTraining: {len(dataset)} samples, {len(dataloader)} batches/epoch, {num_epochs} epochs")
print(f"Total steps: {total_steps}\n")

global_step = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        # batch_size=1 returns list: [{input_ids:..., ...}]
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
        torch.cuda.empty_cache()

        global_step += 1
        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if global_step % 50 == 0:
            model.save_pretrained(os.path.join(OUTPUT, f"checkpoint-{global_step}"))

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# Save final model
model.save_pretrained(OUTPUT)
processor.save_pretrained(OUTPUT)
print(f"\nModel saved to {OUTPUT}")
PYEOF

echo ""
echo "========================================"
echo " 训练完成！"
echo " 模型: $OUTPUT"
echo "========================================"
