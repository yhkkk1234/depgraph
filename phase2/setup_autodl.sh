#!/bin/bash
# ============================================================
# Phase 2: 环境搭建 + 模型下载
# 使用方法: bash setup_autodl.sh
# ============================================================
set -e

WORKDIR="/root/autodl-tmp"
# 自动检测脚本所在目录 (适配不同上传布局)
PHASE2_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$PHASE2_DIR")"
DATA_DIR="$WORKDIR/phase2_data"
MODEL_DIR="$WORKDIR/models"
OUTPUT="$WORKDIR/output_phase2"

mkdir -p "$DATA_DIR" "$MODEL_DIR" "$OUTPUT"

echo "Phase2 脚本目录: $PHASE2_DIR"
echo "共享模块目录:   $PARENT_DIR"

echo "========================================"
echo " Step 0: CUDA 环境检查"
echo "========================================"
python -c "
import sys, subprocess
# 检查 CUDA
try:
    nvcc = subprocess.run(['nvcc','--version'], capture_output=True, text=True)
    print(f'nvcc: {nvcc.stdout.split(chr(10))[-2]}')
except:
    print('nvcc: 未找到 (将使用预编译 wheel)')

# 检查 GPU
try:
    out = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                        capture_output=True, text=True)
    print(f'GPU: {out.stdout.strip()}')
except:
    print('GPU: 无法检测')

print(f'Python: {sys.version.split()[0]}')
" 2>&1 || echo "跳过检查"

echo ""
echo "========================================"
echo " Step 1: pip 国内源 + 安装依赖 (版本锁定)"
echo "========================================"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装 CJK 字体 (修复 matplotlib 中文渲染)
apt-get update -qq && apt-get install -y -qq fonts-wqy-microhei 2>/dev/null || echo "  字体安装跳过 (非 root 或无网络)"
# 清除 matplotlib 字体缓存使新字体生效
rm -rf ~/.cache/matplotlib 2>/dev/null || true

# ═══════════════════════════════════════════
# 版本策略: 检查已有 → 满足就不重装
#  Qwen2-VL-7B 最低要求: torch>=2.0, transformers>=4.43
#  CUDA 12.1 镜像是 torch 2.4.1 (最高可用)
# ═══════════════════════════════════════════
echo "检查已有 PyTorch..."
python -c "
import torch; v=torch.__version__
print(f'  当前: {v}')
major, minor = int(v.split('.')[0]), int(v.split('.')[1])
print('  OK' if minor >= 3 else '  需要升级')
"

if python -c "import torch; v=torch.__version__; exit(0 if int(v.split('.')[1])>=3 else 1)" 2>/dev/null; then
    echo "  PyTorch 版本满足，跳过重装"
else
    echo "  安装 PyTorch 2.4.1 (CUDA 12.1 最高可用)..."
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121 -q
fi

pip install "transformers>=4.46.0,<4.50" -q
pip install "peft>=0.13.0" "accelerate>=0.33.0" -q
pip install "bitsandbytes>=0.43.0" -q
pip install datasets pillow -q
pip install qwen-vl-utils -q
pip install matplotlib networkx huggingface_hub -q
# 注意: 不用 modelscope (需要 Python 3.9+)，改用 huggingface_hub + 国内镜像

echo ""
echo "========================================"
echo " Step 1.5: 验证安装版本"
echo "========================================"
python -c "
import torch, transformers, peft, bitsandbytes
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:   {torch.version.cuda}')
    print(f'GPU count:      {torch.cuda.device_count()}')
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB')
print(f'transformers: {transformers.__version__}')
print(f'peft:         {peft.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
# 版本检查
tv = tuple(int(x) for x in transformers.__version__.split('.')[:2])
assert tv >= (4, 46), f'transformers {transformers.__version__} < 4.46!'
tv2 = tuple(int(x) for x in torch.__version__.split('.')[:2])
assert tv2 >= (2, 3), f'torch {torch.__version__} < 2.3!'
print('All version checks PASSED')
"

echo ""
echo "========================================"
echo " Step 2: 下载 Qwen2-VL-7B-Instruct (HF 镜像)"
echo "========================================"
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-7B-Instruct', cache_dir='$MODEL_DIR')
print('Done.')
"
MODEL_PATH=$(find "$MODEL_DIR" -name "config.json" -path "*Qwen2-VL*" | head -1 | xargs dirname)
echo "Model: $MODEL_PATH"

echo ""
echo "========================================"
echo " Step 3: 生成 Phase 2 训练/评估数据"
echo "========================================"
export PHASE2_DIR
export PARENT_DIR
export DATA_DIR
python << 'PYEOF'
import sys, os

PHASE2_DIR = os.environ["PHASE2_DIR"]
PARENT_DIR = os.environ["PARENT_DIR"]
DATA_DIR = os.environ["DATA_DIR"]

sys.path.insert(0, PHASE2_DIR)
sys.path.insert(0, PARENT_DIR)  # render_diagram 等共享模块

from phase2_data_gen import (
    generate_exp1_data, generate_exp2_data,
    generate_exp3_data, generate_exp4_data
)

print("生成 Exp1 训练数据 (主动画图, 500条)...")
generate_exp1_data(f"{DATA_DIR}/exp1")

print("生成 Exp2 测试数据 (长对话记忆)...")
generate_exp2_data(f"{DATA_DIR}/exp2", round_levels=[50, 100, 200])

print("生成 Exp3 测试数据 (图复杂度)...")
generate_exp3_data(f"{DATA_DIR}/exp3", levels=["simple", "moderate", "complex"])

print("生成 Exp4 测试数据 (零样本迁移)...")
generate_exp4_data(f"{DATA_DIR}/exp4")

print(f"\n全部数据已生成到 {DATA_DIR}")
# 统计
import json, glob
for sub in ["exp1", "exp2", "exp3", "exp4"]:
    files = glob.glob(f"{DATA_DIR}/{sub}/**", recursive=True)
    print(f"  {sub}: {len(files)} 文件")
PYEOF

echo ""
echo "========================================"
echo " 环境搭建完成！"
echo " 数据目录: $DATA_DIR"
echo " 模型路径: $MODEL_PATH"
echo ""
echo " 下一步: bash gen_data.sh  (如需重新生成数据)"
echo "         bash train_phase2.sh"
echo "========================================"
