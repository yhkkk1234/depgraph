#!/bin/bash
# ============================================================
# Phase 2: 生成训练和评估数据
# 在 AutoDL 上运行 (需要先 setup_autodl.sh 安装依赖)
#
# 使用方法: bash gen_data.sh
# ============================================================
set -e

WORKDIR="/root/autodl-tmp"
DATA_DIR="$WORKDIR/phase2_data"
PHASE2_DIR="$WORKDIR/phase2"

mkdir -p "$DATA_DIR"

echo "========================================"
echo " Phase 2 数据生成"
echo "========================================"

python << 'PYEOF'
import sys
sys.path.insert(0, "/root/autodl-tmp/phase2")

from phase2_data_gen import (
    generate_exp1_data,
    generate_exp2_data,
    generate_exp3_data,
    generate_exp4_data,
)

DATA_DIR = "/root/autodl-tmp/phase2_data"

print("Exp1: 主动画图训练数据 (500条, 代码+文章+对话)...")
generate_exp1_data(f"{DATA_DIR}/exp1")

print("\nExp2: 长对话测试数据 (50/100/200轮)...")
generate_exp2_data(f"{DATA_DIR}/exp2", round_levels=[50, 100, 200])

print("\nExp3: 图复杂度测试数据 (简单/中等/复杂)...")
generate_exp3_data(f"{DATA_DIR}/exp3", levels=["simple", "moderate", "complex"])

print("\nExp4: 零样本迁移测试数据...")
generate_exp4_data(f"{DATA_DIR}/exp4")

print(f"\n数据生成完成！输出: {DATA_DIR}")
PYEOF

echo ""
echo " 数据生成完成！"
echo " 输出: $DATA_DIR/exp1 ~ exp4"
echo ""
echo " 下一步: bash train_phase2.sh"
echo "========================================"
