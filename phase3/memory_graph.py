#!/usr/bin/env python
"""
memory_graph — Phase 3 记忆图生成器

将对话转化为持久记忆图，支持:
  - 规则提取模式: 基于关键词+轮次邻近度
  - 增量更新: 新对话合并到已有记忆图
  - 标注管理: 圈注、评语、重要性标记
  - 时间衰减: 旧话题逐渐"冷却"

与 Phase 2 conversation_graph.py 的核心区别:
  - Phase 2: 输出用于导航的话题关系图 (话题名+轮次范围)
  - Phase 3: 输出用作记忆本身的记忆图 (话题+摘要+圈注+评语+时间颜色)

用法:
  python memory_graph.py conversation.json -o ./output/
  python memory_graph.py conversation.json --update-from v3 --mode memory
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

from phase3.render_memory_graph import render_memory_graph, generate_memory_key


# 中文停用词 — 过滤跨域通用词防止记忆图被噪音填满
_CN_STOP_WORDS = {
    "讨论", "需要", "问题", "考虑", "具体", "一些", "这个", "那个",
    "可以", "应该", "非常", "比较", "进行", "使用", "通过", "因为",
    "所以", "但是", "如果", "已经", "还是", "什么", "怎么", "为什么",
    "就是", "另外", "不过", "来说", "的话", "对于", "关于", "之后",
    "之前", "一样", "不同", "以及", "或者", "并且", "而且", "虽然",
    "因此", "然而", "首先", "其次", "最后", "第一", "第二", "第三",
    "方面", "这个问", "个问题", "我们需", "们需要", "需要考", "要考虑",
}


def _tokenize(text: str) -> set[str]:
    tokens = set()
    tokens.update(re.findall(r'[a-zA-Z]{2,}', text.lower()))
    for chinese_block in re.findall(r'[\u4e00-\u9fff]{2,}', text):
        for i in range(len(chinese_block) - 1):
            tok = chinese_block[i:i+2]
            if tok not in _CN_STOP_WORDS:
                tokens.add(tok)
        for i in range(len(chinese_block) - 2):
            tok = chinese_block[i:i+3]
            if tok not in _CN_STOP_WORDS:
                tokens.add(tok)
    tokens.update(re.findall(r'[a-zA-Z0-9_]{2,}', text.lower()))
    return tokens


def build_memory_graph(conversation: list[dict],
                       annotations: dict = None,
                       existing_graph: dict = None,
                       version: int = 1) -> dict:
    """从对话构建/更新记忆图。

    Args:
        conversation: 对话列表 [{"round": 1, "role": "user", "content": "...", ...}, ...]
        annotations: 用户标注 {"round_5": {"circle": True, "comment": "重点关注"}, ...}
        existing_graph: 已有记忆图 dict (增量更新模式)
        version: 图版本号

    Returns:
        记忆图 dict，兼容 render_memory_graph()
    """
    annotations = annotations or {}
    rounds = len(conversation)

    if rounds == 0:
        return _empty_memory_graph(version)

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
        return _empty_memory_graph(version)

    word_freq = defaultdict(int)
    word_rounds = defaultdict(list)
    for r, info in round_keywords.items():
        for w in info["keywords"]:
            word_freq[w] += 1
            word_rounds[w].append(r)

    min_rounds = max(2, min(rounds * 0.06, 3))
    topics_raw = {}
    topic_id = 0
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        if freq >= min_rounds and len(word) >= 2:
            topic_name = f"话题_{word[:10]}"
            first_r = min(word_rounds[word])
            last_r = max(word_rounds[word])
            topics_raw[topic_name] = {
                "keyword": word,
                "first_round": first_r,
                "last_round": last_r,
                "freq": freq,
                "rounds": sorted(word_rounds[word]),
                "time_factor": last_r / rounds if rounds > 0 else 0.5,
            }
            topic_id += 1
            if topic_id >= 30:
                break

    topic_descriptions = {}
    for name, info in topics_raw.items():
        desc_parts = []
        for r in info["rounds"][:3]:
            if r in round_keywords:
                snippet = round_keywords[r]["text"][:60].strip()
                if snippet:
                    desc_parts.append(snippet)
        topic_descriptions[name] = " | ".join(desc_parts) if desc_parts else info["keyword"]

    relationships = []
    topic_names = list(topics_raw.keys())
    for i, name_a in enumerate(topic_names):
        for j, name_b in enumerate(topic_names):
            if i >= j:
                continue
            info_a = topics_raw[name_a]
            info_b = topics_raw[name_b]
            common = set(info_a["rounds"]) & set(info_b["rounds"])
            if not common:
                continue
            r_overlap = min(common)
            if abs(info_a["first_round"] - info_b["first_round"]) <= 3:
                rel_type = "references"
            elif info_a["first_round"] < info_b["first_round"]:
                rel_type = "causes"
            else:
                rel_type = "references"
            relationships.append({
                "from": name_a, "to": name_b,
                "type": rel_type,
                "round": r_overlap,
            })

    imp_map = {"high": 0, "medium": 1, "low": 2, "normal": 3}

    nodes = []
    for name, info in topics_raw.items():
        imp = _calc_importance(info, annotations)
        node_annotations = _extract_annotations(name, info, annotations)
        nodes.append({
            "id": name,
            "label": _make_label(info["keyword"]),
            "description": topic_descriptions.get(name, info["keyword"])[:100],
            "importance": imp,
            "time_factor": info["time_factor"],
            "annotations": node_annotations,
            "round_range": [info["first_round"], info["last_round"]],
        })

    nodes.sort(key=lambda n: (imp_map.get(n["importance"], 3), -n["time_factor"]))

    if existing_graph and existing_graph.get("nodes"):
        nodes = _merge_nodes(existing_graph["nodes"], nodes, rounds)

    # 限制节点数量。增量模式允许更多节点以保留旧记忆
    max_nodes = 45 if existing_graph and existing_graph.get("nodes") else 28
    if len(nodes) > max_nodes:
        top_n = min(max_nodes - 8, len(nodes))
        keep_nodes = nodes[:top_n]
        old_tail = [n for n in nodes[top_n:] if n.get("time_factor", 1) < 0.5]
        keep_nodes.extend(old_tail[:(max_nodes - len(keep_nodes))])
        nodes = keep_nodes

    metadata = {
        "version": version,
        "total_rounds": rounds,
        "title": "对话记忆图",
    }

    return {
        "nodes": nodes,
        "edges": relationships,
        "metadata": metadata,
    }


def _make_label(keyword: str) -> str:
    """生成简短显示标签"""
    keyword = keyword.strip().replace("_", " ")
    if len(keyword) > 10:
        return keyword[:9] + ".."
    return keyword


def _calc_importance(info: dict, annotations: dict) -> str:
    """综合判断话题重要性"""
    for ann_round, ann_data in annotations.items():
        try:
            ar = int(ann_round) if isinstance(ann_round, str) else ann_round
        except (ValueError, TypeError):
            continue
        if ar in info.get("rounds", []):
            if isinstance(ann_data, dict) and ann_data.get("circle"):
                return "high"
            if isinstance(ann_data, dict) and ann_data.get("importance") == "high":
                return "high"
    freq = info.get("freq", 0)
    if freq >= 8:
        return "high"
    elif freq >= 4:
        return "medium"
    return "normal"


def _extract_annotations(name: str, info: dict, annotations: dict) -> list[dict]:
    """提取节点相关的标注"""
    result = []
    for ann_round, ann_data in annotations.items():
        try:
            ar = int(ann_round) if isinstance(ann_round, str) else ann_round
        except (ValueError, TypeError):
            continue
        if ar in info.get("rounds", []):
            if isinstance(ann_data, dict):
                if ann_data.get("circle"):
                    result.append({"type": "circle", "text": ann_data.get("circle_text", "重点关注")})
                if ann_data.get("comment"):
                    result.append({"type": "comment", "text": ann_data["comment"]})
    return result


def _merge_nodes(old_nodes: list[dict], new_nodes: list[dict],
                 total_rounds: int) -> list[dict]:
    """合并旧图节点和新图节点，实现时间衰减。

    三种情况:
      - 全新节点: 保持 build_memory_graph 计算的 time_factor (通常 ≈1.0)
      - 重新出现的旧节点: 混合旧衰减分数 + 新激活分数 (50/50)
      - 消失的旧节点: 纯衰减 (×0.6 每 session)
    """
    old_map = {n["id"]: n for n in old_nodes}
    merged_ids = set()
    result = []

    for nn in new_nodes:
        nid = nn["id"]
        merged_ids.add(nid)
        if nid in old_map:
            old = old_map[nid]
            old_rr = old.get("round_range", [0, 0])
            new_rr = nn.get("round_range", [0, 0])
            nn["round_range"] = [
                min(old_rr[0], new_rr[0]) if old_rr[0] > 0 else new_rr[0],
                max(old_rr[1], new_rr[1]),
            ]
            # 关键修复: 混合旧衰减因子和新激活因子
            old_tf = old.get("time_factor", 0.5)
            new_tf = nn.get("time_factor", 1.0)
            nn["time_factor"] = old_tf * 0.5 + new_tf * 0.5

            old_anns = old.get("annotations", [])
            new_anns = nn.get("annotations", [])
            existing_types = {(a.get("type"), a.get("text")) for a in old_anns}
            for a in new_anns:
                if (a.get("type"), a.get("text")) not in existing_types:
                    old_anns.append(a)
            nn["annotations"] = old_anns
            if old.get("importance") == "high":
                nn["importance"] = max(
                    nn.get("importance", "normal"),
                    "high",
                    key=lambda x: {"high": 0, "medium": 1, "low": 2, "normal": 3}.get(x, 3)
                )
        result.append(nn)

    for old_n in old_nodes:
        nid = old_n["id"]
        if nid not in merged_ids:
            old_n = dict(old_n)
            old_n["time_factor"] = max(0.05, old_n.get("time_factor", 0.5) * 0.6)
            old_imp = old_n.get("importance", "normal")
            if old_imp == "high":
                old_n["importance"] = "medium"
            elif old_imp == "medium":
                old_n["importance"] = "normal"
            result.append(old_n)

    imp_order = {"high": 0, "medium": 1, "low": 2, "normal": 3}
    result.sort(key=lambda n: (imp_order.get(n.get("importance", "normal"), 3),
                                -n.get("time_factor", 0.5)))
    return result


def _empty_memory_graph(version: int = 1) -> dict:
    return {"nodes": [], "edges": [], "metadata": {"version": version, "total_rounds": 0, "title": "空记忆图"}}


def save_memory_graph(graph: dict, output_dir: str, tag: str = "v1"):
    """保存记忆图 (JSON + PNG + Key)"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"memory_graph_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    png_path = out_dir / f"memory_graph_{tag}.png"
    key_path = out_dir / f"memory_key_{tag}.txt"

    if graph.get("nodes"):
        render_memory_graph(graph, output_path=str(png_path))
        with open(key_path, "w", encoding="utf-8") as f:
            f.write(generate_memory_key(graph))

    return str(json_path), str(png_path) if graph.get("nodes") else None, str(key_path)


def main():
    parser = argparse.ArgumentParser(description="对话→记忆图生成器 (Phase 3)")
    parser.add_argument("conversation", help="对话 JSON 文件路径")
    parser.add_argument("--output", "-o", default="./phase3_output/", help="输出目录")
    parser.add_argument("--update-from", "-u", help="从已有图版本增量更新 (如: v3)")
    parser.add_argument("--version", "-v", type=int, default=0, help="版本号 (0=自动检测)")
    args = parser.parse_args()

    conv_path = Path(args.conversation)
    if not conv_path.exists():
        print(f"错误: 文件不存在: {conv_path}", file=sys.stderr)
        sys.exit(1)

    with open(conv_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    conversation = data.get("conversation", data) if isinstance(data, dict) else data
    annotations = data.get("annotations", {}) if isinstance(data, dict) else {}

    out_dir = Path(args.output)

    existing_graph = None
    if args.update_from:
        existing_path = out_dir / f"memory_graph_{args.update_from}.json"
        if existing_path.exists():
            with open(existing_path, "r", encoding="utf-8") as f:
                existing_graph = json.load(f)
            print(f"增量更新: 基于 {args.update_from}", file=sys.stderr)
        else:
            print(f"警告: 版本 {args.update_from} 不存在，重新生成", file=sys.stderr)

    existing_versions = sorted(out_dir.glob("memory_graph_v*.json"))
    version = args.version if args.version > 0 else (len(existing_versions) + 1)
    version_tag = f"v{version}"

    graph = build_memory_graph(conversation, annotations, existing_graph, version)
    results = save_memory_graph(graph, str(out_dir), version_tag)

    print(f"JSON: {results[0]}", file=sys.stderr)
    if results[1]:
        print(f"PNG:  {results[1]}", file=sys.stderr)
    print(f"Key:  {results[2]}", file=sys.stderr)
    print(f"节点: {len(graph.get('nodes', []))} | 边: {len(graph.get('edges', []))} "
          f"| 版本: {version_tag}", file=sys.stderr)


if __name__ == "__main__":
    main()
