#!/usr/bin/env python
"""
render_memory_graph — Phase 3 记忆图专用渲染器

将记忆图数据渲染为视觉记忆图 (PNG)，支持:
  - 时间远近颜色 (暖→冷)
  - 重要性节点大小
  - 圈注效果 (⭐ + 粗红边框)
  - 评语旁注
  - 关系类型区分 (实线/虚线/红色双向/细线+)
  - 图例说明

与 Phase 1/2 render_diagram.py 的区别:
  - 独立配色方案 (时间维度而非模块类型)
  - 节点标注 (星标 + 评语旁注)
  - 边类型多样化
  - 时间图例 + 标注图例
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

_CJK_FONT = None
for _f in fm.fontManager.ttflist:
    if _f.name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
                   "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                   "Noto Sans CJK", "Noto Sans SC"):
        _CJK_FONT = _f.name
        break
if _CJK_FONT:
    plt.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

STAR_MARKER = "\u2605"

# 时间颜色映射: 0.0(旧) → 冷蓝灰, 1.0(新) → 暖橙红
def time_color(factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    r = 0.40 + factor * 0.55
    g = 0.55 + factor * 0.20 - (factor - 0.5) * 0.33 if factor > 0.5 else 0.55 + factor * 0.20
    b = 0.82 - factor * 0.55
    g = max(0.0, min(1.0, g))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

NODE_SIZE_MAP = {"high": 2800, "medium": 2000, "low": 1400, "normal": 1600}
BORDER_WIDTH_MAP = {"high": 2.5, "medium": 1.5, "low": 1.0, "normal": 1.0}


def render_memory_graph(graph: dict, output_path: str = None,
                        as_data_url: bool = False) -> str:
    """渲染记忆图到 PNG。

    graph 格式:
    {
      "nodes": [
        {
          "id": "topic_1",
          "label": "支付模块架构",
          "description": "简短摘要",
          "importance": "high|medium|low",
          "time_factor": 0.9,
          "annotations": [
            {"type": "circle", "text": "重点关注"},
            {"type": "comment", "text": "用户认为重试机制可能雪崩"}
          ],
          "round_range": [1, 15]
        }, ...
      ],
      "edges": [
        {"from": "topic_1", "to": "topic_2", "type": "causes", "label": ""}, ...
      ],
      "metadata": {"version": 1, "total_rounds": 50, "title": "..."}
    }
    """
    nodes = graph.get("nodes", [])
    edges_list = graph.get("edges", [])
    metadata = graph.get("metadata", {})

    if not nodes:
        raise ValueError("记忆图为空: 无节点")

    G = nx.DiGraph()
    node_labels = {}
    node_sizes = []
    node_colors = []
    node_border_colors = []
    node_border_widths = []
    circle_nodes = set()
    node_comments = {}
    node_importances = {}

    for n in nodes:
        nid = n["id"]
        label = n.get("label", nid)
        imp = n.get("importance", "normal")
        tf = n.get("time_factor", 0.5)
        G.add_node(nid)
        node_labels[nid] = label
        node_sizes.append(NODE_SIZE_MAP.get(imp, 1600))
        node_colors.append(time_color(tf))
        node_importances[nid] = imp

        has_circle = False
        comments = []
        for ann in n.get("annotations", []):
            if ann.get("type") == "circle":
                has_circle = True
            elif ann.get("type") == "comment":
                comments.append(ann.get("text", ""))
        if has_circle:
            circle_nodes.add(nid)
            node_border_colors.append("#DC143C")
            node_border_widths.append(3.0)
        else:
            node_border_colors.append("#495057")
            node_border_widths.append(BORDER_WIDTH_MAP.get(imp, 1.0))
        if comments:
            node_comments[nid] = comments

    for e in edges_list:
        src, dst = e["from"], e["to"]
        if src in G.nodes() and dst in G.nodes():
            etype = e.get("type", "references")
            G.add_edge(src, dst, type=etype)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=130)
    pos = nx.spring_layout(G, k=3.0, iterations=80, seed=42)

    edge_styles = {
        "causes": ("solid", "#4A4A4A", 1.5, True),
        "references": ("dashed", "#888888", 1.2, True),
        "contradicts": ("solid", "#CC0000", 1.8, False),
        "extends": ("dotted", "#666666", 1.0, True),
    }

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors,
        edgecolors=node_border_colors, linewidths=node_border_widths,
        alpha=0.92, ax=ax
    )

    for etype, (style, color, width, arrow) in edge_styles.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == etype]
        if edges_of_type:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_of_type, edge_color=color,
                style=style, width=width, arrows=arrow, arrowsize=14,
                connectionstyle="arc3,rad=0.12", alpha=0.7, ax=ax
            )

    for nid, (x, y) in pos.items():
        lbl = node_labels.get(nid, nid)
        imp = node_importances.get(nid, "normal")
        font_size = 8 if imp == "high" else 7
        font_weight = "bold" if imp == "high" else "normal"
        ax.text(x, y, lbl, fontsize=font_size, fontweight=font_weight,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.75))

    for nid in circle_nodes:
        if nid in pos:
            x, y = pos[nid]
            ax.text(x, y + 0.06, STAR_MARKER, fontsize=14, color="#DC143C",
                    ha="center", va="bottom", fontweight="bold",
                    transform=ax.transData)

    for nid, comments in node_comments.items():
        if nid in pos and comments:
            x, y = pos[nid]
            comment_text = comments[0][:25]
            if len(comments[0]) > 25:
                comment_text += "..."
            ax.annotate(
                comment_text,
                xy=(x, y), xytext=(x + 0.06, y - 0.09),
                fontsize=6, color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFDE7",
                          edgecolor="#CCCCCC", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.8),
                ha="left", va="top",
            )

    legend_elements = [
        mpatches.Patch(facecolor=time_color(1.0), edgecolor="#999", linewidth=1,
                        label=f"\u65b0\u8fd1\u671f ({time_color(1.0)})"),
        mpatches.Patch(facecolor=time_color(0.5), edgecolor="#999", linewidth=1,
                        label=f"\u4e2d\u671f ({time_color(0.5)})"),
        mpatches.Patch(facecolor=time_color(0.1), edgecolor="#999", linewidth=1,
                        label=f"\u65e9\u671f ({time_color(0.1)})"),
        mpatches.Patch(facecolor="white", edgecolor="#DC143C", linewidth=3,
                        label=f"{STAR_MARKER} \u7528\u6237\u5708\u6ce8\u91cd\u70b9"),
    ]

    line_handles = [
        plt.Line2D([0], [0], color="#4A4A4A", linestyle="solid", lw=1.5,
                    label="\u56e0\u679c (causes)"),
        plt.Line2D([0], [0], color="#888888", linestyle="dashed", lw=1.2,
                    label="\u5f15\u7528 (references)"),
        plt.Line2D([0], [0], color="#CC0000", linestyle="solid", lw=1.8,
                    label="\u77db\u76fe (contradicts)"),
        plt.Line2D([0], [0], color="#666666", linestyle="dotted", lw=1.0,
                    label="\u5ef6\u4f38 (extends)"),
    ]

    all_legend = legend_elements + line_handles
    ax.legend(handles=all_legend, loc="lower left", fontsize=7,
              framealpha=0.9, ncol=2, title="\u56fe\u4f8b", title_fontsize=8)

    title = metadata.get("title", "\u8bb0\u5fc6\u56fe")
    version = metadata.get("version", 1)
    total_rounds = metadata.get("total_rounds", 0)
    full_title = f"{title} (v{version}, {total_rounds}\u8f6e\u5bf9\u8bdd)"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=15)

    ax.axis("off")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    plt.tight_layout(pad=2.0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(buf.read())
        return output_path

    buf.seek(0)
    raw_b64 = base64.b64encode(buf.read()).decode()
    if as_data_url:
        return f"data:image/png;base64,{raw_b64}"
    return raw_b64


def generate_memory_key(graph: dict) -> str:
    """生成记忆图文字 Key（补充 OCR 不足）。

    格式：
      [★ 重点关注] 话题名 (轮次 1-15)
        摘要: ...
        评语: ...
        关联: → 话题B, ← 话题C
    """
    nodes = graph.get("nodes", [])
    edges_list = graph.get("edges", [])
    if not nodes:
        return "# 记忆图 — 空前\n"

    lines = ["# 记忆图文字说明\n"]
    metadata = graph.get("metadata", {})
    lines.append(f"版本 v{metadata.get('version', 1)} | "
                 f"共 {len(nodes)} 个话题节点 | "
                 f"{metadata.get('total_rounds', 0)} 轮对话\n")

    adj_out = {}
    adj_in = {}
    for e in edges_list:
        src, dst = e["from"], e["to"]
        adj_out.setdefault(src, []).append(dst)
        adj_in.setdefault(dst, []).append(src)

    imp_order = {"high": 0, "medium": 1, "low": 2, "normal": 3}
    sorted_nodes = sorted(nodes, key=lambda n: (
        imp_order.get(n.get("importance", "normal"), 3),
        -(n.get("time_factor", 0.5))
    ))

    for n in sorted_nodes:
        nid = n["id"]
        label = n.get("label", nid)
        imp = n.get("importance", "normal")
        rr = n.get("round_range", [0, 0])
        desc = n.get("description", "")
        tf = n.get("time_factor", 0.5)

        imp_icon = {"high": STAR_MARKER, "medium": "\u25cf", "low": "\u25cb", "normal": " "}.get(imp, " ")
        imp_cn = {"high": "重点关注", "medium": "重要", "low": "一般", "normal": ""}.get(imp, "")

        lines.append(f"\n{imp_icon} [{imp_cn}] **{label}** (话题 {nid}, "
                     f"轮次 {rr[0]}-{rr[1]}, 时间系数 {tf:.1f})")
        if desc:
            lines.append(f"  摘要: {desc}")

        for ann in n.get("annotations", []):
            if ann.get("type") == "circle":
                lines.append(f"  \u270f\ufe0f 圈注: {ann.get('text', '')}")
            elif ann.get("type") == "comment":
                lines.append(f"  \U0001f4dd 评语: {ann.get('text', '')}")

        if nid in adj_out:
            lines.append(f"  \u2192 关联到: {', '.join(adj_out[nid][:5])}")
        if nid in adj_in:
            lines.append(f"  \u2190 被引用: {', '.join(adj_in[nid][:5])}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python render_memory_graph.py <graph.json> [output.png]", file=sys.stderr)
        sys.exit(1)

    graph_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "memory_graph.png"

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    result = render_memory_graph(graph, output_path=output_path)
    print(f"渲染完成: {result}")

    key_path = output_path.replace(".png", "_key.txt")
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(generate_memory_key(graph))
    print(f"文字Key: {key_path}")
