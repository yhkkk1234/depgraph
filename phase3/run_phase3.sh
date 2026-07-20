#!/bin/bash
# ============================================================
#  Phase 3 完整实验运行脚本
#  用法: bash run_phase3.sh [options]
#
#  Options:
#    --data-only   仅生成数据, 不评估
#    --eval-only   仅评估, 不重新生成数据
#    --full        生成全部数据并评估 (默认)
#    --samples N   设置数据生成样本数 (默认: 30/20/500/50)
#    -h            帮助
# ============================================================
set -e

# ── 默认参数 ──
MODE="full"
N_EXP1=30
N_EXP3=20
N_EXP5=500
N_EXP6=50
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "用法: bash run_phase3.sh [options]"
    echo ""
    echo "Options:"
    echo "  --data-only     仅生成实验数据"
    echo "  --eval-only     仅运行评估 (不重新生成数据)"
    echo "  --full          生成数据 + 评估 (默认)"
    echo "  --samples N     设置各实验样本数"
    echo "  -h              显示帮助"
    exit 0
}

# ── 解析参数 ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only) MODE="data" ;;
        --eval-only) MODE="eval" ;;
        --full)      MODE="full" ;;
        --samples)   N_EXP1=$2; N_EXP3=$2; N_EXP5=$2; N_EXP6=$2; shift ;;
        -h)          usage ;;
        *)           echo "Unknown: $1"; usage ;;
    esac
    shift
done

# ── 切换到 experiment/ (phase3 父目录) ──
cd "$PARENT_DIR"
echo "============================================"
echo " Phase 3 Run Script"
echo " Working dir: $(pwd)"
echo " Mode: $MODE"
echo "============================================"

# ── 确保输出目录存在 ──
mkdir -p phase3_output/data
mkdir -p phase3_output/eval_results

# ── 数据生成 ──
if [ "$MODE" = "data" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "[1/2] Generating experimental data..."
    echo "  Exp1 samples: $N_EXP1"
    echo "  Exp3 samples: $N_EXP3"
    echo "  Exp5 samples: $N_EXP5"
    echo "  Exp6 samples: $N_EXP6"
    echo ""

    python -c "
import sys; sys.path.insert(0, '.')
from phase3.phase3_data_gen import gen_exp1, gen_exp2, gen_exp3, gen_exp4, gen_exp5, gen_exp6

out = 'phase3_output/data'
print('=== Exp1: Static Memory Graphs ===')
gen_exp1(out, $N_EXP1)
print()
print('=== Exp2: Incremental Memory ===')
gen_exp2(out)
print()
print('=== Exp3: Annotation Ablation ===')
gen_exp3(out, $N_EXP3)
print()
print('=== Exp4: Scale Expansion ===')
gen_exp4(out)
print()
print('=== Exp5: Training Data ===')
gen_exp5(out, $N_EXP5)
print()
print('=== Exp6: Zero-Shot Transfer ===')
gen_exp6(out, $N_EXP6)
print()
print('Data generation complete!')
"

    echo "  Data generation: DONE"
fi

# ── 评估 ──
if [ "$MODE" = "eval" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "[2/2] Running evaluation..."
    python -c "
import sys; sys.path.insert(0, '.')
from phase3.phase3_eval import eval_exp1, eval_exp2, eval_exp3, eval_exp4, eval_exp5, eval_exp6
import json, os, time

out_dir = 'phase3_output/eval_results'
os.makedirs(out_dir, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
data_dir = 'phase3_output/data'

results = {}
try:
    r1 = eval_exp1(f'{data_dir}/exp1_static/exp1_static_memory.json')
    results['exp1'] = r1
    with open(f'{out_dir}/exp1_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r1, f, ensure_ascii=False)
    print('  Exp1: OK')
except Exception as e: print(f'  Exp1 FAIL: {e}')

try:
    r2 = eval_exp2(f'{data_dir}/exp2_incremental/exp2_incremental_memory.json')
    results['exp2'] = r2
    with open(f'{out_dir}/exp2_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r2, f, ensure_ascii=False)
    print('  Exp2: OK')
except Exception as e: print(f'  Exp2 FAIL: {e}')

try:
    r3 = eval_exp3(f'{data_dir}/exp3_ablation/exp3_ablation.json')
    results['exp3'] = r3
    with open(f'{out_dir}/exp3_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r3, f, ensure_ascii=False)
    print('  Exp3: OK')
except Exception as e: print(f'  Exp3 FAIL: {e}')

try:
    r4 = eval_exp4(f'{data_dir}/exp4_scale/exp4_scale_all.json')
    results['exp4'] = r4
    with open(f'{out_dir}/exp4_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r4, f, ensure_ascii=False)
    print('  Exp4: OK')
except Exception as e: print(f'  Exp4 FAIL: {e}')

try:
    r5 = eval_exp5(f'{data_dir}/exp5_train/exp5_train_data.json')
    results['exp5'] = r5
    with open(f'{out_dir}/exp5_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r5, f, ensure_ascii=False)
    print('  Exp5: OK')
except Exception as e: print(f'  Exp5 FAIL: {e}')

try:
    r6 = eval_exp6(f'{data_dir}/exp6_transfer/exp6_transfer.json')
    results['exp6'] = r6
    with open(f'{out_dir}/exp6_{ts}.json', 'w', encoding='utf-8') as f: json.dump(r6, f, ensure_ascii=False)
    print('  Exp6: OK')
except Exception as e: print(f'  Exp6 FAIL: {e}')

with open(f'{out_dir}/phase3_report_{ts}.json', 'w', encoding='utf-8') as f:
    json.dump({'phase': 'Phase 3', 'timestamp': ts, 'experiments': results}, f, ensure_ascii=False)
print('  Report saved.')
"
    echo "  Evaluation: DONE"
fi

echo ""
echo "============================================"
echo " Phase 3 completed!"
echo "  Data:    $(pwd)/phase3_output/data/"
echo "  Results: $(pwd)/phase3_output/eval_results/"
echo "============================================"
