#!/bin/bash
# ============================================================
#  Phase 3 AutoDL 一键部署脚本
#  适用: AutoDL 4090/4090D (24GB), CUDA 12.x
#
#  用法:
#    chmod +x setup_phase3.sh
#    bash setup_phase3.sh
# ============================================================
set -e

echo "============================================"
echo " Phase 3: Graphical Memory System AutoDL Setup"
echo "============================================"
echo ""

# ── 1. 基础依赖 ──
echo "[1/4] Installing Python packages..."
pip install matplotlib networkx numpy -q
echo "  Done."

# ── 2. CJK 字体 (matplotlib 中文渲染) ──
echo "[2/4] Installing CJK fonts..."
if ! fc-list | grep -qi "simhei\|yahei\|noto.*cjk\|wenquanyi" 2>/dev/null; then
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq fonts-wqy-microhei 2>/dev/null && echo "  Installed WQY Micro Hei" || {
        # 备选: 下载 Noto Sans CJK
        FONT_DIR="$HOME/.fonts"
        mkdir -p "$FONT_DIR"
        wget -q "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf" \
            -O "$FONT_DIR/NotoSansCJKsc-Regular.otf" 2>/dev/null && \
            fc-cache -f && echo "  Installed Noto Sans CJK SC" || \
            echo "  WARNING: CJK fonts not installed - Chinese labels may render as boxes"
    }
else
    echo "  CJK fonts already installed."
fi

python -c "
import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    if any(k in f.name for k in ['SimHei', 'YaHei', 'Noto Sans CJK', 'WenQuanYi', 'Micro Hei']):
        print(f'  Available CJK font: {f.name}')
        break
" || echo "  (font check skipped)"

# ── 3. 验证目录结构 ──
echo "[3/4] Verifying directory structure..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "  Script dir: $SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/render_memory_graph.py" ]; then
    echo "  ERROR: phase3 source files not found!"
    echo "  Make sure to upload all .py files to the same directory as this script."
    exit 1
fi
echo "  Source files verified."

# ── 4. 快速冒烟测试 ──
echo "[4/4] Running smoke test..."
cd "$SCRIPT_DIR/.."
python -c "
import sys, os
sys.path.insert(0, '.')
from phase3.memory_graph import build_memory_graph
from phase3.render_memory_graph import render_memory_graph
print('  Phase3 modules import OK')
"
echo "  Smoke test passed!"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Generate all data:  python phase3_data_gen.py --all -o ./phase3_data/"
echo "   2. Run evaluation:     python phase3_eval.py --all -d ./phase3_data/ -o ./eval_results/"
echo "   3. Test memory graph:  python memory_graph.py test_conv.json -o ./output/"
echo "============================================"
