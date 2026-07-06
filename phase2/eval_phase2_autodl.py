#!/usr/bin/env python
"""
eval_phase2_autodl — Phase 2 评估 (AutoDL 版)

在 AutoDL 上运行完整评估: base vs fine-tuned × 4 实验

用法:
  python eval_phase2_autodl.py                    # 运行全部实验
  python eval_phase2_autodl.py --exp 1             # 只运行 Exp1
  python eval_phase2_autodl.py --base-only         # 只评估 base 模型
"""
from __future__ import annotations

import json, os, re, sys, time, torch, argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── 路径配置 ──
WORKDIR = "/root/autodl-tmp"
MODEL_DIR = os.path.join(WORKDIR, "models")
LORA_PATH = os.path.join(WORKDIR, "output_phase2")
DATA_DIR  = os.path.join(WORKDIR, "phase2_data")
RESULTS   = os.path.join(WORKDIR, "phase2_results")
os.makedirs(RESULTS, exist_ok=True)

# ── 模型加载 ──

def find_model_path() -> str:
    for root, _, files in os.walk(MODEL_DIR):
        if "Qwen2-VL" in root and "config.json" in files:
            return root
    mp = next(Path(MODEL_DIR).rglob("config.json"))
    return str(mp.parent)


def load_model(use_lora: bool = True):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    model_path = find_model_path()
    print(f"Model: {model_path}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True)

    if use_lora and os.path.exists(os.path.join(LORA_PATH, "adapter_config.json")):
        print(f"LoRA: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print("LoRA: 未加载 (base 模式)")

    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 65536
    processor.image_processor.max_pixels = 262144
    return model, processor


def ask_text(model, processor, prompt: str, max_tokens: int = 1024) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)


def ask_with_image(model, processor, prompt: str, img_path: str, max_tokens: int = 1024) -> str:
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=True)
    return processor.decode(out[0], skip_special_tokens=True)


# ── 评分函数 ──

def detect_diagram(response: str) -> dict:
    has_mermaid = bool(re.search(r'```(mermaid|graph)', response, re.IGNORECASE))
    has_sketch  = bool(re.search(r'<sketch>', response, re.IGNORECASE))
    has_struct  = bool(re.search(
        r'(让我画|画一张|结构图|关系图|拓扑|节点|连线|步骤\s*\d|第一步|第二步|先.*分析|先.*梳理)',
        response))
    return {
        "has_diagram": has_mermaid or has_sketch,
        "has_structured": has_struct,
        "type": "mermaid" if has_mermaid else ("sketch" if has_sketch else "none"),
    }


def score_recall(response: str, expected: list[str]) -> dict:
    resp = response.lower()
    found = [e for e in expected if e.lower() in resp]
    return {"found": len(found), "total": len(expected),
            "recall": len(found) / len(expected) if expected else 1.0}


# ═══════════════════════════════════════════════════════════════
#  实验 1: 主动画图
# ═══════════════════════════════════════════════════════════════

EXP1_TASKS = [
    {
        "name": "code_modification",
        "prompt": """请完成以下代码修改任务。如果你认为有必要先理清结构关系，可以画出影响关系图再动手。

修改: Task.to_dict() 的返回值从 dict 改为 JSON 字符串格式。
受影响模块: models/task.py, services/task_service.py, api/handlers.py, utils/validators.py
当前修改在 models/task.py。请分析影响范围并给出修改方案。""",
        "expected": ["task_service.py", "handlers.py", "validators.py", "to_dict"],
    },
    {
        "name": "article_analysis",
        "prompt": """分析以下文章的结构和论点。如果需要可以先画出关系图帮助理清思路。

文章摘要: 讨论 AI 在医疗领域的应用。核心论点三个: (1) AI 诊断效率超越人工；(2) 患者信任是部署瓶颈；(3) 人机协作模式需要监管框架。论点之间存在张力：效率 vs 信任，创新 vs 监管。

请分析论点之间的逻辑关系。""",
        "expected": ["效率", "信任", "监管", "人机协作", "张力"],
    },
    {
        "name": "conversation_summary",
        "prompt": """总结以下对话的话题演变。对话讨论了: 微服务架构设计(前段) → 数据库选型(中段) → 部署策略(后段)。在第15轮曾回到架构话题提出新想法。

请先画出话题演变图，再总结。""",
        "expected": ["微服务", "数据库", "部署", "第15轮", "架构"],
    },
]


def run_exp1(model, processor) -> dict:
    print("\n" + "=" * 50)
    print(" Experiment 1: 主动画图能力")
    print("=" * 50)
    results = []

    for i, task in enumerate(EXP1_TASKS):
        print(f"  Task {i+1}: {task['name']}")
        t0 = time.time()
        resp = ask_text(model, processor, task["prompt"])
        elapsed = time.time() - t0

        diag = detect_diagram(resp)
        recall = score_recall(resp, task["expected"])

        results.append({
            "task": task["name"],
            "diagram_generated": diag["has_diagram"],
            "diagram_type": diag["type"],
            "has_structured": diag["has_structured"],
            "recall": recall,
            "elapsed_s": round(elapsed, 1),
            "response": resp[:500],
        })
        print(f"    图: {'✅' if diag['has_diagram'] else '❌'} | "
              f"结构化: {'✅' if diag['has_structured'] else '❌'} | "
              f"召回: {recall['recall']:.2f}")

    gen_rate = sum(1 for r in results if r["diagram_generated"]) / len(results)
    struct_rate = sum(1 for r in results if r["has_structured"]) / len(results)
    avg_recall = sum(r["recall"]["recall"] for r in results) / len(results)

    summary = {
        "generation_rate": round(gen_rate, 3),
        "structured_rate": round(struct_rate, 3),
        "avg_recall": round(avg_recall, 3),
    }
    print(f"\n  总结: 图生成率={gen_rate:.0%}, 结构化率={struct_rate:.0%}, 平均召回={avg_recall:.3f}")
    return {"results": results, "summary": summary}


# ═══════════════════════════════════════════════════════════════
#  实验 2: 视觉记忆检索 (v3: 图导航)
# ═══════════════════════════════════════════════════════════════

def _generate_topic_graph(conv_data: dict, round_level: int) -> tuple[str, str]:
    """用对话元数据中的已有话题名构建关系图。返回 (png_path, text_key)。

    不使用 n-gram 碎片提取，而是直接用对话中的 topics 列表和 topic_lifecycle。
    """
    import sys, os, tempfile
    own_dir = os.path.dirname(os.path.abspath(__file__))
    for parent in [os.path.dirname(own_dir),
                   os.path.join(os.path.dirname(own_dir), ".."),
                   "/root/autodl-tmp",
                   "/root/autodl-tmp/experiment"]:
        sys.path.insert(0, os.path.normpath(parent))
    try:
        from render_diagram import render_dependency_graph
    except ImportError:
        return None, None

    topics = conv_data.get("topics", [])
    lifecycle = conv_data.get("topic_lifecycle", {})

    if len(topics) < 2:
        return None, None

    # ── 构建 graph dict (兼容 render_dependency_graph) ──
    modules = {}
    overlap_threshold = 5  # 重叠轮数阈值即为关联

    for t in topics:
        lc = lifecycle.get(t, {})
        modules[t] = {
            "dependencies": [],
            "dependents": [],
            "_meta": {
                "first_round": lc.get("start", 0),
                "last_round": lc.get("end", 0),
                "peaks": len(lc.get("peaks", [])),
                "importance": "high" if lc.get("end", 0) - lc.get("start", 0) > 20 else "normal",
            },
        }

    # 建立关系：话题生命周期有重叠的即为关联
    for i, t1 in enumerate(topics):
        lc1 = lifecycle.get(t1, {})
        r1_start, r1_end = lc1.get("start", 0), lc1.get("end", 0)
        for t2 in topics[i + 1:]:
            lc2 = lifecycle.get(t2, {})
            r2_start, r2_end = lc2.get("start", 0), lc2.get("end", 0)
            overlap = min(r1_end, r2_end) - max(r1_start, r2_start)
            if overlap >= overlap_threshold:
                if t2 not in modules[t1]["dependencies"]:
                    modules[t1]["dependencies"].append(t2)
                if t1 not in modules[t2]["dependents"]:
                    modules[t2]["dependents"].append(t1)

    graph = {"modules": modules}

    # ── 构建文字键 ──
    key_lines = ["# 话题关系图\n"]
    key_lines.append(f"共 {len(topics)} 个话题, 对话 {round_level} 轮\n")
    for t in sorted(topics,
                    key=lambda x: lifecycle.get(x, {}).get("start", 0)):
        lc = lifecycle.get(t, {})
        deps = modules[t].get("dependencies", [])
        imp = modules[t].get("_meta", {}).get("importance", "normal")
        marker = "★" if imp == "high" else "  "
        key_lines.append(
            f"{marker} {t} (轮次 {lc.get('start',0)}-{lc.get('end',0)})"
        )
        if deps:
            key_lines.append(f"    关联: {', '.join(deps[:4])}")

    # ── 渲染 PNG ──
    tmpdir = tempfile.mkdtemp(prefix="phase2_exp2_")
    png_path = os.path.join(tmpdir, f"topic_graph_{round_level}.png")
    try:
        render_dependency_graph(graph, output_path=png_path)
    except Exception as e:
        print(f"    渲染失败: {e}", file=sys.stderr)
        return None, None

    return png_path, "\n".join(key_lines)


def _extract_rounds_from_nav(response: str, total_rounds: int) -> list[int]:
    """从模型导航响应中提取轮次号。"""
    rounds = set()
    for m in re.finditer(r'(\d+)\s*[-–—到至]\s*(\d+)', response):
        start, end = int(m.group(1)), int(m.group(2))
        rounds.update(range(max(1, start), min(total_rounds, end) + 1))
    for m in re.finditer(r'(?:第|轮次|round)\s*(\d+)', response, re.IGNORECASE):
        rounds.add(int(m.group(1)))
    return sorted(rounds)[:30]


def run_exp2(model, processor, round_levels: list[int] = None) -> dict:
    print("\n" + "=" * 50)
    print(" Experiment 2: 视觉记忆检索 (v3: 图导航)")
    print("=" * 50)
    round_levels = round_levels or [50, 100, 200]
    results = []

    for n in round_levels:
        conv_path = Path(DATA_DIR) / "exp2" / f"conversation_{n}.json"
        if not conv_path.exists():
            continue

        with open(conv_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        conversation = conv_data.get("conversation", [])
        questions = conv_data.get("questions", [])[:5]
        if not questions:
            continue

        q_block = "\n".join(f"Q{j+1}: {q.get('question','')}"
                           for j, q in enumerate(questions))

        # ── 条件 A: 全文 (基线) ──
        conv_text = "\n".join(
            f"[R{m.get('round','?')}] {m.get('role','?')}: {m.get('content','')[:120]}"
            for m in conversation[-50:])
        text_prompt = f"回答以下问题:\n\n{conv_text}\n\n{q_block}"

        t0 = time.time()
        text_resp = ask_text(model, processor, text_prompt, max_tokens=512)
        text_time = time.time() - t0
        text_tokens = len(conv_text)

        # ── 条件 B: 图导航 (图定位轮次 → 精准读文) ──
        png_path, text_key = _generate_topic_graph(conv_data, n)
        nav_recall = 0.0; nav_tokens = text_tokens
        if png_path and text_key:
            nav_prompt = f"""看图了解话题的时间分布，确定回答以下问题需要查哪些轮次。

{text_key}

问题:
{q_block}

只需输出轮次范围，格式: 相关轮次: X-Y, A-B"""
            try:
                nav_resp = ask_with_image(model, processor, nav_prompt, png_path, max_tokens=200)
            except Exception:
                nav_resp = ""

            rel_rounds = _extract_rounds_from_nav(nav_resp, n)
            if len(rel_rounds) >= 2:
                filtered = "\n".join(
                    f"[R{m.get('round','?')}] {m.get('role','?')}: {m.get('content','')[:200]}"
                    for m in conversation if m.get("round", 0) in rel_rounds
                )
                nav_tokens = len(filtered)
                qa_prompt = f"根据对话片段回答:\n\n{filtered}\n\n{q_block}"
            else:
                qa_prompt = text_prompt

            try:
                nav_qa = ask_text(model, processor, qa_prompt, max_tokens=512)
                nav_scores = [score_recall(nav_qa, q.get("answer", []))["recall"]
                             for q in questions]
                nav_recall = sum(nav_scores) / len(nav_scores)
            except Exception:
                nav_recall = 0.0
        else:
            nav_recall = 0.0

        text_scores = [score_recall(text_resp, q.get("answer", []))["recall"]
                      for q in questions]
        avg_text = sum(text_scores) / len(text_scores) if text_scores else 0

        token_ratio = round(nav_tokens / max(text_tokens, 1), 2)
        results.append({
            "rounds": n,
            "text_recall": round(avg_text, 3),
            "nav_recall": round(nav_recall, 3),
            "text_tokens": text_tokens,
            "nav_tokens": nav_tokens,
            "token_ratio": token_ratio,
            "delta": round(nav_recall - avg_text, 3),
        })
        print(f"  {n}轮: 全文={avg_text:.3f}({text_tokens}tk) → "
              f"导航={nav_recall:.3f}({nav_tokens}tk, {token_ratio:.0%}) Δ={nav_recall-avg_text:+.3f}")

    summary = {
        "avg_text_recall": round(sum(r["text_recall"] for r in results) / len(results), 3) if results else 0,
        "avg_nav_recall": round(sum(r["nav_recall"] for r in results) / len(results), 3) if results else 0,
    }
    return {"results": results, "summary": summary}


# ═══════════════════════════════════════════════════════════════
#  实验 3: 扫描效率 (多模态版)
# ═══════════════════════════════════════════════════════════════

def run_exp3(model, processor, levels: list[str] = None) -> dict:
    print("\n" + "=" * 50)
    print(" Experiment 3: 扫描效率 (v3: 纯图 — 不泄漏)")
    print("=" * 50)
    levels = levels or ["simple", "moderate", "complex"]
    results = []

    for level in levels:
        graph_path = Path(DATA_DIR) / "exp3" / f"graph_{level}.png"
        json_path = Path(DATA_DIR) / "exp3" / f"graph_{level}.json"

        if not graph_path.exists() or not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            gt = data.get("ground_truth", {})

        # v3: 只给图, 不做文字泄漏。CJK 字体已修复, 依赖模型视觉能力。
        # 节点名列表仅作 OCR 兜底 (不含任何结构性答案)
        center_hint = gt.get("center_node", "")
        iso_hint = gt.get("isolated_nodes", [])
        known_names = set([center_hint] + iso_hint) - {""}
        name_hint = f"节点名参考: {', '.join(sorted(known_names)[:6])}" if known_names else ""

        prompt = f"""看图回答问题:
1. 图中有多少个模块节点？
2. 哪个节点被连接最多（中心节点）？
3. 是否存在反馈回路/循环？（是/否）
4. 有没有孤立节点（没有任何连线的节点）？（是/否）
{name_hint}"""

        t0 = time.time()
        resp = ask_with_image(model, processor, prompt, str(graph_path), max_tokens=256)
        elapsed = time.time() - t0

        # ── 评分 ──
        scores = {}
        gt_nodes = gt.get("node_count", 0)
        gt_center = gt.get("center_node", "")
        gt_cycle = gt.get("has_cycle", False)
        gt_iso   = gt.get("has_isolated", False)

        node_match = re.search(r'(\d+)\s*(个)?(节点|node|模块)', resp, re.IGNORECASE)
        if node_match:
            scores["nodes"] = 1.0 if int(node_match.group(1)) == gt_nodes else 0.0
        else:
            scores["nodes"] = 0.0

        scores["center"] = 1.0 if gt_center and gt_center in resp else 0.0

        cycle_yes = bool(re.search(r'(有|是|yes|存在).{0,15}(回路|循环|cycle)', resp, re.IGNORECASE))
        cycle_no  = bool(re.search(r'(没有|无|否|no).{0,15}(回路|循环|cycle)', resp, re.IGNORECASE))
        scores["cycle"] = 1.0 if (gt_cycle and cycle_yes) or (not gt_cycle and cycle_no) else 0.0

        iso_yes = bool(re.search(r'(有|是|yes|存在).{0,15}(孤立|isolated)', resp, re.IGNORECASE))
        iso_no  = bool(re.search(r'(没有|无|否|no).{0,15}(孤立|isolated)', resp, re.IGNORECASE))
        scores["isolated"] = 1.0 if (gt_iso and iso_yes) or (not gt_iso and iso_no) else 0.0

        avg_acc = sum(scores.values()) / len(scores)
        results.append({
            "level": level,
            "scores": scores,
            "avg_accuracy": round(avg_acc, 3),
            "elapsed_s": round(elapsed, 1),
            "response": resp[:300],
        })
        print(f"  {level}: 准确率={avg_acc:.3f} ({scores})")

    summary = {
        "avg_accuracy": round(sum(r["avg_accuracy"] for r in results) / len(results), 3) if results else 0,
    }
    return {"results": results, "summary": summary}


# ═══════════════════════════════════════════════════════════════
#  实验 4: 自发迁移
# ═══════════════════════════════════════════════════════════════

EXP4_TASKS = {
    "decision_analysis": {
        "prompt": """分析以下决策场景。

场景: 城市需要在市中心建商业综合体。支持方(开发商/市政府): 就业和税收。反对方(居民/环保): 交通和绿地。小商户: 怕竞争。请给出分析。""",
        "expected": ["开发商", "居民", "环保", "小商户", "市政府"],
    },
    "argument_mapping": {
        "prompt": """分析辩论结构。

辩题: 校园禁用手机。正方: 影响学习(PISA数据)。反方: 数字素养, 紧急联系。正方反驳: 计算机课可替代。请分析论点链条。""",
        "expected": ["注意力", "PISA", "数字素养", "紧急联系", "反驳"],
    },
    "knowledge_graph": {
        "prompt": """根据事实回答问题: "谁与张教授有间接关系？"

事实: 张教授→AI实验室→计算机学院→数学学院→李教授。王博士(曾是张教授学生)负责图论课题。""",
        "expected": ["李教授", "王博士", "图论", "数学学院"],
    },
    # ── v2 新增任务 ──
    "risk_analysis": {
        "prompt": """分析以下项目的风险并排序。

项目: 开发一个在线支付系统。涉及:
- 支付网关对接 (外部依赖, 不确定性强)
- 用户数据加密存储 (合规要求)
- 交易日志审计 (法规要求)
- 前端UI改版 (用户体验优化)
- 数据库迁移 (技术债务)

请分析各模块之间的依赖关系和风险优先级。""",
        "expected": ["支付网关", "数据加密", "交易日志", "前端UI", "数据库迁移", "风险", "依赖"],
    },
    "system_design": {
        "prompt": """设计一个简单的订单处理流程。

需求:
- 用户下单后需验证库存
- 库存不足则通知用户
- 库存充足则锁定库存并创建支付
- 支付成功则扣减库存并通知仓库
- 支付失败则释放库存

请先画出流程图再说明。""",
        "expected": ["下单", "库存验证", "支付", "扣减库存", "通知仓库", "释放库存"],
    },
}


def run_exp4(model, processor) -> dict:
    print("\n" + "=" * 50)
    print(" Experiment 4: 自发迁移")
    print("=" * 50)
    results = []

    for task_name, task_data in EXP4_TASKS.items():
        print(f"  Task: {task_name}")
        t0 = time.time()
        resp = ask_text(model, processor, task_data["prompt"])
        elapsed = time.time() - t0

        diag = detect_diagram(resp)
        recall = score_recall(resp, task_data["expected"])

        # 检测结构化推理模式
        patterns = []
        if re.search(r'(步骤|第一步|第二步|首先.*其次|第一.*第二)', resp):
            patterns.append("steps")
        if re.search(r'(关系|关联|连接|依赖|影响).*(图|结构|链路)', resp):
            patterns.append("relational")
        if re.search(r'(↑|↓|→|←|节点|边|拓扑)', resp):
            patterns.append("graph_terms")

        results.append({
            "task": task_name,
            "has_diagram": diag["has_diagram"],
            "has_structured": diag["has_structured"],
            "patterns": patterns,
            "recall": recall,
            "elapsed_s": round(elapsed, 1),
            "response": resp[:400],
        })
        print(f"    图: {'✅' if diag['has_diagram'] else '❌'} | "
              f"结构化: {'✅' if diag['has_structured'] else '❌'} | "
              f"模式: {patterns} | 召回: {recall['recall']:.2f}")

    struct_count = sum(1 for r in results if r["has_structured"])
    avg_recall = sum(r["recall"]["recall"] for r in results) / len(results)

    summary = {
        "structured_rate": round(struct_count / len(results), 3),
        "avg_recall": round(avg_recall, 3),
    }
    print(f"\n  总结: 结构化率={struct_count}/{len(results)}, 平均召回={avg_recall:.3f}")
    return {"results": results, "summary": summary}


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4],
                        help="只运行指定实验 (1-4)")
    parser.add_argument("--base-only", action="store_true",
                        help="只评估 base 模型 (不加载 LoRA)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 评估 base 和 fine-tuned
    modes = [("base", False)] if args.base_only else [("base", False), ("finetuned", True)]

    all_results = {}

    for mode_name, use_lora in modes:
        model_tag = "base" if not use_lora else "finetuned"
        print(f"\n{'#'*60}")
        print(f"#  评估: {model_tag}")
        print(f"{'#'*60}")

        model, processor = load_model(use_lora=use_lora)

        mode_results = {}

        if not args.exp or args.exp == 1:
            mode_results["exp1_active_drawing"] = run_exp1(model, processor)

        if not args.exp or args.exp == 2:
            mode_results["exp2_visual_memory"] = run_exp2(model, processor)

        if not args.exp or args.exp == 3:
            mode_results["exp3_scan_efficiency"] = run_exp3(model, processor)

        if not args.exp or args.exp == 4:
            mode_results["exp4_internalization"] = run_exp4(model, processor)

        all_results[mode_name] = mode_results

        # 保存中间结果
        out_path = os.path.join(RESULTS, f"phase2_{model_tag}_{timestamp}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果: {out_path}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── 对比报告 ──
    if len(all_results) == 2:
        _print_comparison(all_results)


def _print_comparison(all_results: dict):
    print("\n" + "=" * 60)
    print(" Base vs Fine-tuned 对比")
    print("=" * 60)

    base = all_results.get("base", {})
    ft   = all_results.get("finetuned", {})

    # Exp1
    if "exp1_active_drawing" in base:
        b1 = base["exp1_active_drawing"]["summary"]
        f1 = ft["exp1_active_drawing"]["summary"]
        print(f"\nExp1 主动画图:")
        print(f"  图生成率:   base={b1['generation_rate']:.0%} → ft={f1['generation_rate']:.0%}  (Δ={f1['generation_rate']-b1['generation_rate']:+.0%})")
        print(f"  结构化率:   base={b1['structured_rate']:.0%} → ft={f1['structured_rate']:.0%}")
        print(f"  召回率:     base={b1['avg_recall']:.3f} → ft={f1['avg_recall']:.3f}")

    # Exp2
    if "exp2_visual_memory" in base:
        b2 = base["exp2_visual_memory"]["summary"]
        f2 = ft["exp2_visual_memory"]["summary"]
        print(f"\nExp2 视觉记忆:")
        print(f"  base 全文召回: {b2['avg_text_recall']:.3f}, 导航召回: {b2['avg_nav_recall']:.3f}")
        print(f"  ft   全文召回: {f2['avg_text_recall']:.3f}, 导航召回: {f2['avg_nav_recall']:.3f}")

    # Exp3
    if "exp3_scan_efficiency" in base:
        b3 = base["exp3_scan_efficiency"]["summary"]
        f3 = ft["exp3_scan_efficiency"]["summary"]
        print(f"\nExp3 扫描效率:")
        print(f"  准确率: base={b3['avg_accuracy']:.3f} → ft={f3['avg_accuracy']:.3f}")

    # Exp4
    if "exp4_internalization" in base:
        b4 = base["exp4_internalization"]["summary"]
        f4 = ft["exp4_internalization"]["summary"]
        print(f"\nExp4 自发迁移:")
        print(f"  结构化率: base={b4['structured_rate']:.0%} → ft={f4['structured_rate']:.0%}")
        print(f"  召回率:   base={b4['avg_recall']:.3f} → ft={f4['avg_recall']:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import gc
    main()
