#!/bin/bash
# ============================================================
# Phase 2: Qwen2-VL-7B 微调 — 主动画图 + 结构化思维
# 
# 训练目标: 让模型学会在推理中主动输出 <sketch> 图示
# 数据域: 代码修改 + 文章分析 + 对话总结
# 
# 使用方法: bash train_phase2.sh
# ============================================================
set -e

WORKDIR="/root/autodl-tmp"
PHASE2_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$PHASE2_DIR")"
DATA_DIR="$WORKDIR/phase2_data"
MODEL_DIR="$WORKDIR/models"
OUTPUT="$WORKDIR/output_phase2"
TRAIN_FORMAT="$WORKDIR/phase2_train_format"

mkdir -p "$OUTPUT" "$TRAIN_FORMAT"

echo "========================================"
echo " Phase 2 训练: 主动画图习惯"
echo "========================================"

echo ""
echo "========================================"
echo " Step 0: 版本确认"
echo "========================================"
python -c "
import torch, transformers
print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用！'
tv = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert tv >= (4, 46), f'transformers {transformers.__version__} < 4.46!'
print('Version check PASSED')
"

echo ""
echo "========================================"
echo " Step 1: 转换训练数据格式"
echo "========================================"
export DATA_DIR TRAIN_FORMAT
python << 'PYEOF'
import json, os, glob
DATA_DIR = os.environ["DATA_DIR"]
TRAIN_FORMAT = os.environ["TRAIN_FORMAT"]
os.makedirs(TRAIN_FORMAT, exist_ok=True)

# 收集所有训练样本
samples = []

# Exp1 数据 (主动画图训练 — 这是我们微调的核心)
exp1_files = glob.glob(f"{DATA_DIR}/exp1/**/*.jsonl", recursive=True)
for f in exp1_files:
    with open(f, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

print(f"加载 {len(samples)} 条训练样本")

# 转换为 Qwen2-VL 对话格式
formatted = []
for s in samples:
    msgs = []
    # 系统提示
    if s.get("system_prompt"):
        msgs.append({"role": "system", "content": s["system_prompt"]})
    # 用户消息
    msgs.append({"role": "user", "content": s["user"]})
    # 助手回复 (包含 <sketch> + <analysis> 块)
    msgs.append({"role": "assistant", "content": s["assistant"]})

    formatted.append({
        "messages": msgs,
        "task_type": s.get("task_type", "unknown"),
        "id": s.get("id", ""),
    })

# 保存
output_path = os.path.join(TRAIN_FORMAT, "dataset.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"保存 {len(formatted)} 条训练样本到 {output_path}")

# 统计各任务类型
from collections import Counter
task_counts = Counter(s.get("task_type") for s in samples)
for task, count in task_counts.items():
    print(f"  {task}: {count}")
PYEOF

echo ""
echo "========================================"
echo " Step 2: 加载模型并开始训练"
echo "========================================"

export MODEL_DIR TRAIN_FORMAT OUTPUT PARENT_DIR
python << 'PYEOF'
import json, os, torch, gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

gc.collect()
torch.cuda.empty_cache()

MODEL_DIR = os.environ["MODEL_DIR"]
TRAIN_FORMAT = os.environ["TRAIN_FORMAT"]
OUTPUT = os.environ["OUTPUT"]
os.makedirs(OUTPUT, exist_ok=True)

# ── 加载模型 (Qwen2-VL-7B, 4-bit QLoRA) ──
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_PATH = None
for root, dirs, files in os.walk(MODEL_DIR):
    if "Qwen2-VL" in root and "config.json" in files:
        MODEL_PATH = root
        break

if not MODEL_PATH:
    # fallback
    mp = next(Path(MODEL_DIR).rglob("config.json"))
    MODEL_PATH = str(mp.parent)

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

# ── 加载数据集 ──
print("\nLoading Phase 2 training dataset...")
with open(os.path.join(TRAIN_FORMAT, "dataset.jsonl")) as f:
    samples = [json.loads(line) for line in f if line.strip()]

print(f"Total training samples: {len(samples)}")

# Phase 2 专用 Dataset: 处理 <sketch> 块中的文本图示
class Phase2Dataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        msgs = s["messages"]

        # 构建聊天格式 (纯文本，<sketch> 块是文本内容)
        chat = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            chat.append({"role": role, "content": content})

        text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text], images=None,
            return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


dataset = Phase2Dataset(samples, processor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ── 训练配置 ──
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 7
total_steps = len(dataloader) * num_epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"\nTraining: {len(dataset)} samples, {len(dataloader)} batches/epoch")
print(f"Epochs: {num_epochs}, Total steps: {total_steps}\n")

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
        torch.cuda.empty_cache()

        global_step += 1
        epoch_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

        if global_step % 50 == 0:
            ckpt = os.path.join(OUTPUT, f"checkpoint-{global_step}")
            model.save_pretrained(ckpt)
            print(f"\n  Checkpoint saved: {ckpt}")

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

# 保存最终模型
model.save_pretrained(OUTPUT)
processor.save_pretrained(OUTPUT)
print(f"\nModel saved to {OUTPUT}")
print("Phase 2 训练完成!")
PYEOF

echo ""
echo "========================================"
echo " Phase 2 微调完成！"
echo " 输出模型: $OUTPUT"
echo ""
echo " 下一步: python eval_phase2_autodl.py"
echo "========================================"
