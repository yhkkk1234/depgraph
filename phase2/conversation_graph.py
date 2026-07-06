#!/usr/bin/env python
"""
conversation_graph — 对话→话题关系图转换器

将对话历史转化为话题关系图，输出兼容 render_diagram.py 的 graph dict。
支持 LLM 提取（精确）和规则提取（快速）两种模式。
生成: 视觉图 + 话题文字键 + 标注文件

用法:
  python conversation_graph.py conversation.json -o ./output/
  python conversation_graph.py conversation.json --update-from v3 --mode llm
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from render_diagram import render_dependency_graph


# ═══════════════════════════════════════════════════════════════
#  规则提取模式：基于关键词 + 轮次邻近度构建话题图
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set[str]:
    """轻量分词：提取有意义的词元。

    英文: 2字母以上单词
    中文: 2-3字 n-gram (覆盖常见中文词组如"微服务""模块化""依赖性")
    """
    tokens = set()
    # 英文词
    tokens.update(re.findall(r'[a-zA-Z]{2,}', text.lower()))
    # 中文: 提取连续中文字符段，生成 2-gram 和 3-gram
    for chinese_block in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        # 2-gram: "微服务架构" -> "微服", "服务", "务架", "架构"
        for i in range(len(chinese_block) - 1):
            tokens.add(chinese_block[i:i+2])
        # 3-gram: "微服务架构" -> "微服务", "服务架", "务架构"
        for i in range(len(chinese_block) - 2):
            tokens.add(chinese_block[i:i+3])
    # 也保留纯数字/混合标识符
    tokens.update(re.findall(r'[a-zA-Z0-9_]{2,}', text.lower()))
    return tokens


def extract_topics_rule(conversation: list[dict], annotations: dict = None) -> dict:
    """基于规则从对话中提取话题和关系。

    策略：
    1. 用实体关键词作为话题种子
    2. 轮次邻近的话题建立 'references' 边
    3. 跨轮次重复出现的话题建立 'extends' 边
    4. 标注文件中的重要性传递到节点
    """
    annotations = annotations or {}
    rounds = len(conversation)

    # ── 收集每轮的关键词 ──
    round_keywords = {}
    for msg in conversation:
        r = msg.get("round", 0)
        text = msg.get("content", "")
        role = msg.get("role", "user")
        keywords = _tokenize(text)
        if r not in round_keywords:
            round_keywords[r] = {"keywords": set(), "text": "", "role": role}
        round_keywords[r]["keywords"] |= keywords
        round_keywords[r]["text"] += text + " "

    if not round_keywords:
        return _empty_graph()

    # ── 提取话题节点 ──
    # 话题 = 出现超过阈值的关键词 + round 区间
    word_freq = defaultdict(int)
    word_rounds = defaultdict(list)
    for r, info in round_keywords.items():
        for w in info["keywords"]:
            word_freq[w] += 1
            word_rounds[w].append(r)

    min_rounds = max(2, min(rounds * 0.08, 4))  # 话题至少出现 2 轮或 8% 轮次，上限 4
    topics = {}
    topic_id = 0
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_rounds and len(word) >= 2:
            topic_name = f"话题_{word[:8]}"
            topics[topic_name] = {
                "keyword": word,
                "first_round": min(word_rounds[word]),
                "last_round": max(word_rounds[word]),
                "freq": freq,
                "rounds": sorted(word_rounds[word]),
            }
            topic_id += 1
            if topic_id >= 25:  # 限制话题数防止图过密
                break

    # ── 构建关系 ──
    modules = {}
    for name, info in topics.items():
        deps = []
        # 与轮次上邻近的话题建立关系
        for other_name, other_info in topics.items():
            if other_name == name:
                continue
            # 跨话题关系：两个话题在同一轮出现 -> references
            common = set(info["rounds"]) & set(other_info["rounds"])
            if common:
                deps.append(other_name)
        modules[name] = {
            "dependencies": deps,
            "dependents": [],
            "_meta": {
                "keyword": info["keyword"],
                "first_round": info["first_round"],
                "last_round": info["last_round"],
                "freq": info["freq"],
                "importance": _get_importance(name, info, annotations),
            },
        }

    # 计算反向依赖
    _compute_dependents(modules)

    return {"modules": modules, "edges": {}, "mode": "rule"}


def _get_importance(topic_name: str, info: dict, annotations: dict) -> str:
    """根据标注和频率判断话题重要性"""
    for ann_round, ann_data in annotations.items():
        if isinstance(ann_round, int) and ann_round in info["rounds"]:
            if ann_data.get("importance") == "high":
                return "high"
    if info["freq"] >= 8:
        return "high"
    elif info["freq"] >= 4:
        return "medium"
    return "normal"


# ═══════════════════════════════════════════════════════════════
#  LLM 提取模式：调用多模态/文本模型进行关系抽取
# ═══════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """你是一个对话分析器。请分析以下对话，提取话题和关系。

输出严格的JSON格式:
{
  "topics": [
    {
      "name": "话题简短名称（中文，不超过8个字）",
      "description": "话题简述",
      "first_round": 数字,
      "last_round": 数字,
      "importance": "high|medium|normal",
      "keywords": ["关键词1", "关键词2"]
    }
  ],
  "relationships": [
    {
      "from": "话题A",
      "to": "话题B",
      "type": "causes|references|contradicts|extends",
      "round": 数字,
      "reason": "建立此关系的原因"
    }
  ]
}

关系类型定义:
- causes: A 引出了/导致了 B 的讨论
- references: A 中提到了/引用了 B
- contradicts: B 中的观点与 A 矛盾
- extends: B 是 A 的延伸/细化

--- 对话内容 ---
{conversation_text}
--- 结束 ---
"""


def extract_topics_llm(conversation: list[dict], api_key: str = None,
                       api_base: str = None, model: str = None) -> dict:
    """使用 LLM 从对话中提取话题和关系。

    需要设置环境变量: EXPERIMENT_API_KEY, EXPERIMENT_API_BASE, EXPERIMENT_MODEL
    或通过参数传入。
    """
    api_key = api_key or os.environ.get("EXPERIMENT_API_KEY", "")
    api_base = api_base or os.environ.get("EXPERIMENT_API_BASE", "https://api.example.com/v1")
    model = model or os.environ.get("EXPERIMENT_MODEL", "model-name")

    if not api_key or api_key == "your-key-here":
        print("警告: 未配置 API，回退到规则提取模式", file=sys.stderr)
        return extract_topics_rule(conversation)

    conversation_text = _format_conversation(conversation)
    prompt = EXTRACTION_PROMPT.format(conversation_text=conversation_text)

    try:
        import requests
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        result_text = resp.json()["choices"][0]["message"]["content"]

        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if not json_match:
            raise ValueError("LLM 输出中未找到 JSON")
        llm_result = json.loads(json_match.group())
    except Exception as e:
        print(f"LLM 提取失败: {e}，回退到规则模式", file=sys.stderr)
        return extract_topics_rule(conversation)

    # ── 转换为标准 graph dict ──
    modules = {}
    topic_list = llm_result.get("topics", [])
    relationships = llm_result.get("relationships", [])

    for t in topic_list:
        name = t["name"]
        modules[name] = {
            "dependencies": [],
            "dependents": [],
            "_meta": {
                "description": t.get("description", ""),
                "first_round": t.get("first_round", 0),
                "last_round": t.get("last_round", 0),
                "importance": t.get("importance", "normal"),
                "keywords": t.get("keywords", []),
            },
        }

    for rel in relationships:
        from_t = rel["from"]
        to_t = rel["to"]
        if from_t in modules and to_t in modules:
            if to_t not in modules[from_t]["dependencies"]:
                modules[from_t]["dependencies"].append(to_t)

    _compute_dependents(modules)

    return {"modules": modules, "edges": relationships, "mode": "llm"}


# ═══════════════════════════════════════════════════════════════
#  共享工具函数
# ═══════════════════════════════════════════════════════════════

def _compute_dependents(modules: dict):
    """填充 dependents（反向依赖）"""
    for name in modules:
        modules[name]["dependents"] = []
    for name, info in modules.items():
        for dep in info.get("dependencies", []):
            if dep in modules and name not in modules[dep]["dependents"]:
                modules[dep]["dependents"].append(name)


def _empty_graph() -> dict:
    return {"modules": {}, "edges": {}, "mode": "rule"}


def _format_conversation(conversation: list[dict], max_rounds: int = 60) -> str:
    """格式化对话为文本，截断超长对话"""
    lines = []
    for msg in conversation[-max_rounds:]:
        role = msg.get("role", "unknown")
        r = msg.get("round", "?")
        content = msg.get("content", "")
        lines.append(f"[Round {r}] {role}: {content[:300]}")
    return "\n\n".join(lines)


def generate_topic_key(graph: dict) -> str:
    """生成话题文字键，供 OCR 受限的小模型使用。"""
    modules = graph.get("modules", {})
    if not modules:
        return "# 话题关系图 — 无话题\n"

    lines = ["# 话题关系图 — 文字键\n"]
    lines.append(f"共 {len(modules)} 个话题节点\n")

    # 按重要性排序
    importance_order = {"high": 0, "medium": 1, "normal": 2}
    sorted_topics = sorted(
        modules.items(),
        key=lambda x: (importance_order.get(x[1].get("_meta", {}).get("importance", "normal"), 2), x[0])
    )

    for name, info in sorted_topics:
        meta = info.get("_meta", {})
        imp = meta.get("importance", "normal")
        imp_marker = {"high": "★", "medium": "●", "normal": "○"}.get(imp, "○")
        desc = meta.get("description", meta.get("keyword", name))
        deps = info.get("dependencies", [])
        rdeps = info.get("dependents", [])

        lines.append(f"\n{imp_marker} **{name}** ({imp})")
        lines.append(f"  {desc}")
        if deps:
            lines.append(f"  关联到: {', '.join(deps[:5])}")
        if rdeps:
            lines.append(f"  被引用: {', '.join(rdeps[:5])}")

    return "\n".join(lines)


def generate_minimal_key(graph: dict, highlight: str = None) -> str:
    """生成极简版文字键（3-5行），模拟 Phase 1 的 minimal text key。"""
    modules = graph.get("modules", {})
    if not modules:
        return ""

    high_importance = []
    medium_importance = []

    for name, info in modules.items():
        meta = info.get("_meta", {})
        imp = meta.get("importance", "normal")
        if imp == "high":
            high_importance.append(name)
        elif imp == "medium":
            medium_importance.append(name)

    lines = []
    if highlight and highlight in modules:
        hl_deps = modules[highlight].get("dependencies", [])
        hl_rdeps = modules[highlight].get("dependents", [])
        lines.append(f"当前焦点: {highlight}")
        if hl_rdeps:
            lines.append(f"  依赖方: {', '.join(hl_rdeps[:4])}")
        if hl_deps:
            lines.append(f"  被依赖: {', '.join(hl_deps[:4])}")

    if high_importance:
        lines.append(f"核心话题(★): {', '.join(high_importance[:5])}")
    if medium_importance:
        lines.append(f"重要话题(●): {', '.join(medium_importance[:5])}")

    return "\n".join(lines) if lines else f"话题数: {len(modules)}"


# ═══════════════════════════════════════════════════════════════
#  增量更新
# ═══════════════════════════════════════════════════════════════

def update_graph(existing_graph: dict, new_conversation: list[dict],
                 mode: str = "rule", **kwargs) -> dict:
    """增量更新：解析新对话部分，合并到现有图中。

    策略：
    - 新话题：直接添加
    - 已有话题：更新 last_round 和 freq
    - 旧话题无新活动：保留但降低重要性
    """
    new_graph = extract_topics_rule(new_conversation) if mode == "rule" \
        else extract_topics_llm(new_conversation, **kwargs)

    old_modules = existing_graph.get("modules", {})
    new_modules = new_graph.get("modules", {})

    merged = {}
    for name, info in old_modules.items():
        merged[name] = info
        # 无新活动的旧话题降级
        if name not in new_modules:
            old_imp = info.get("_meta", {}).get("importance", "normal")
            if old_imp == "high":
                info["_meta"]["importance"] = "medium"
            elif old_imp == "medium":
                info["_meta"]["importance"] = "normal"

    for name, info in new_modules.items():
        if name in merged:
            # 更新已有话题
            old_meta = merged[name].get("_meta", {})
            new_meta = info.get("_meta", {})
            old_meta["last_round"] = max(
                old_meta.get("last_round", 0),
                new_meta.get("last_round", 0)
            )
            old_meta["freq"] = old_meta.get("freq", 0) + new_meta.get("freq", 0)
            # 合并依赖
            for dep in info.get("dependencies", []):
                if dep not in merged[name]["dependencies"]:
                    merged[name]["dependencies"].append(dep)
        else:
            merged[name] = info

    _compute_dependents(merged)
    return {"modules": merged, "edges": new_graph.get("edges", {}), "mode": mode}


# ═══════════════════════════════════════════════════════════════
#  标注管理
# ═══════════════════════════════════════════════════════════════

def load_annotations(anno_path: str) -> dict:
    """加载用户标注文件。"""
    if not os.path.exists(anno_path):
        return {}
    with open(anno_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(annotations: dict, anno_path: str):
    """保存标注到文件。"""
    Path(anno_path).parent.mkdir(parents=True, exist_ok=True)
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="对话→话题关系图 — 将对话历史转化为可视化记忆图谱"
    )
    parser.add_argument("conversation", help="对话 JSON 文件路径")
    parser.add_argument("--output", "-o", default="./output/",
                        help="输出目录 (默认: ./output/)")
    parser.add_argument("--highlight", "-H",
                        help="高亮的当前话题")
    parser.add_argument("--mode", "-m", choices=["rule", "llm"], default="rule",
                        help="提取模式: rule=规则提取, llm=LLM提取 (需要API)")
    parser.add_argument("--update-from", "-u",
                        help="从已有图版本增量更新 (如: v3)")
    parser.add_argument("--annotations", "-a",
                        help="用户标注文件路径 (.json)")
    parser.add_argument("--inject", "-i", action="store_true",
                        help="输出 prompt-ready 片段到 stdout")
    args = parser.parse_args()

    # ── 加载对话 ──
    conv_path = Path(args.conversation)
    if not conv_path.exists():
        print(f"错误: 文件不存在: {conv_path}", file=sys.stderr)
        sys.exit(1)

    with open(conv_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    conversation = data.get("conversation", data) if isinstance(data, dict) else data
    annotations = data.get("annotations", {}) if isinstance(data, dict) else {}

    # 合并外部标注文件
    if args.annotations:
        annotations.update(load_annotations(args.annotations))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 版本管理 ──
    existing_versions = sorted(out_dir.glob("topic_graph_v*.json"))
    if args.update_from:
        existing_path = out_dir / f"topic_graph_{args.update_from}.json"
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                existing_graph = json.load(f)
            print(f"增量更新: 基于 {args.update_from}", file=sys.stderr)
            graph = update_graph(
                existing_graph, conversation,
                mode=args.mode,
                api_key=os.environ.get("EXPERIMENT_API_KEY"),
                api_base=os.environ.get("EXPERIMENT_API_BASE"),
                model=os.environ.get("EXPERIMENT_MODEL"),
            )
        else:
            print(f"警告: 版本 {args.update_from} 不存在，重新生成", file=sys.stderr)
            graph = _run_extraction(conversation, annotations, args)
    else:
        graph = _run_extraction(conversation, annotations, args)

    version = len(existing_versions) + 1
    version_tag = f"v{version}"

    # ── 保存图 JSON ──
    graph_json_path = out_dir / f"topic_graph_{version_tag}.json"
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"保存: {graph_json_path}", file=sys.stderr)

    # ── 渲染 PNG ──
    modules = graph.get("modules", {})
    if modules:
        png_path = out_dir / f"topic_graph_{version_tag}.png"
        render_dependency_graph(
            graph,
            highlight_module=args.highlight,
            output_path=str(png_path),
        )
        print(f"渲染: {png_path}", file=sys.stderr)
    else:
        print("警告: 未提取到话题，跳过渲染", file=sys.stderr)

    # ── 生成文字键 ──
    if modules:
        key_path = out_dir / f"topic_key_{version_tag}.txt"
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(generate_topic_key(graph))
        print(f"文字键: {key_path}", file=sys.stderr)

        minimal_key_path = out_dir / f"topic_key_minimal_{version_tag}.txt"
        with open(minimal_key_path, "w", encoding="utf-8") as f:
            f.write(generate_minimal_key(graph, args.highlight))

        # 生成 prompt snippet
        snippet_path = out_dir / f"prompt_snippet_{version_tag}.txt"
        _write_prompt_snippet(snippet_path, graph, args.highlight, version_tag, out_dir)

    # ── 保存标注 ──
    if annotations:
        anno_out = out_dir / f"annotations_{version_tag}.json"
        save_annotations(annotations, str(anno_out))

    # ── inject 模式 ──
    if args.inject and modules:
        snippet = _build_prompt_text(graph, args.highlight)
        print(snippet)

    # ── 统计输出 ──
    topic_count = len(modules)
    edge_count = sum(len(info.get("dependencies", [])) for info in modules.values())
    high_count = sum(
        1 for info in modules.values()
        if info.get("_meta", {}).get("importance") == "high"
    )
    print(f"\n话题: {topic_count} | 关系: {edge_count} | 核心话题: {high_count}", file=sys.stderr)
    print(f"版本: {version_tag} | 模式: {graph.get('mode', 'rule')}", file=sys.stderr)


def _run_extraction(conversation, annotations, args) -> dict:
    if args.mode == "llm":
        return extract_topics_llm(
            conversation,
            api_key=os.environ.get("EXPERIMENT_API_KEY"),
            api_base=os.environ.get("EXPERIMENT_API_BASE"),
            model=os.environ.get("EXPERIMENT_MODEL"),
        )
    return extract_topics_rule(conversation, annotations)


def _write_prompt_snippet(path: Path, graph: dict, highlight: str,
                          version: str, out_dir: Path):
    text = _build_prompt_text(graph, highlight)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _build_prompt_text(graph: dict, highlight: str = None) -> str:
    """构建可注入 LLM 的完整提示片段。"""
    parts = []
    parts.append("## 对话话题关系图\n")
    parts.append("以下是当前对话的话题结构图，请在回答前先浏览此图获取全局视角。\n")
    parts.append(generate_minimal_key(graph, highlight))
    parts.append(f"\n[图片: topic_graph_{_current_version()}.png]")
    parts.append(generate_topic_key(graph))
    return "\n".join(parts)


def _current_version() -> str:
    return "v1"


if __name__ == "__main__":
    main()
