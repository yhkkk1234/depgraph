#!/bin/bash
# ============================================================
#  Phase 3b-1 AutoDL Training Script (Python 3.8+ 兼容)
#  镜像: NVIDIA/cuda-samples/CUDA12.1-Torch2.4.1-Python3.8.20-Transformers4.46
#  训练模型学会主动生成 memory_sketch (Mermaid格式记忆图)
#
#  用法:
#    bash train_phase3.sh                    # 默认 Qwen2.5-7B-Instruct
#    MODEL=Qwen/Qwen2.5-3B-Instruct bash train_phase3.sh  # 小模型快测
# ============================================================
set -e

echo "============================================"
echo " Phase 3b-1: Memory Sketch Training"
echo "============================================"
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3}GB')"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/phase3_output/training/phase3_memory"
OUTPUT_DIR="$SCRIPT_DIR/phase3_output/training/phase3_lora"
mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL_NAME"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# ── 1. 安装依赖 ──
echo ""
echo "[1/4] Installing dependencies..."
pip install -q datasets peft bitsandbytes accelerate fire aiohttp xxhash sentencepiece tokenizers 2>&1 | tail -3
# 确保 bitsandbytes 版本足够新
pip install -q bitsandbytes --upgrade 2>/dev/null || true
python -c "
import peft, bitsandbytes, accelerate, datasets, fire
print('  peft:', peft.__version__)
print('  accelerate:', accelerate.__version__)
print('  bitsandbytes:', bitsandbytes.__version__)
print('  datasets:', datasets.__version__)
"
echo "  Done."

# ── 2. 加载训练数据 ──
echo "[2/4] Loading training data..."
python -c "
import json
with open('$DATA_DIR/phase3_memory_train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)
with open('$DATA_DIR/phase3_memory_val.json', 'r', encoding='utf-8') as f:
    val = json.load(f)
print(f'  Train: {len(train)} samples, Val: {len(val)} samples')
# 检查格式
s = train[0]['conversations']
print(f'  Format: {len(s)} turns, roles: {s[0][\"from\"]}->{s[1][\"from\"]}')
"

# ── 3. 训练 ──
echo "[3/4] Starting QLoRA training (5 epochs, ~40 min on 4090D)..."
echo ""

# CUDA 内存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$SCRIPT_DIR/train_memory_sketch.py" \
    --model_name "$MODEL_NAME" \
    --train_data "$DATA_DIR/phase3_memory_train.json" \
    --val_data "$DATA_DIR/phase3_memory_val.json" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank 8 --lora_alpha 16 \
    --batch_size 1 --gradient_accumulation 8 \
    --learning_rate 5e-5 --num_epochs 5 \
    --max_length 1536

# ── 4. 移出环境创建信息(防报错)>>>
echo ""
echo "[4/4] Verifying LoRA weights..."
python -c "
import json, os
path = '$OUTPUT_DIR'
files = os.listdir(path) if os.path.exists(path) else []
print(f'  Files: {files}')
adapter = os.path.join(path, 'adapter_model.safetensors')
if os.path.exists(adapter):
    import os; print(f'  adapter_model.safetensors: {os.path.getsize(adapter)//1024}KB')
    print('  Training successful!')
else:
    print('  WARNING: adapter_model.safetensors not found')
"

echo ""
echo "============================================"
echo " Training complete!"
echo " LoRA weights: $OUTPUT_DIR"
echo ""
echo " Test command:"
echo "   python test_memory_sketch.py --lora $OUTPUT_DIR --model $MODEL_NAME"
echo "============================================"
