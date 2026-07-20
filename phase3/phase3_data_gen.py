#!/usr/bin/env python
"""
phase3_data_gen — Phase 3 实验数据生成器

生成 4+2 个实验所需的全部数据:
  Phase 3a (概念验证, 0训练):
    --exp1: 静态记忆图 — "扫图即回忆" (30段对话 → 记忆图 + QA)
    --exp2: 增量记忆积累 — "越积越厚的笔记本" (5段连续对话，多版本图)
    --exp3: 标注效果消融 — 圈注/评语/高亮的作用 (4版对比)
    --exp4: 规模扩展 — 信息量-优势曲线 (4个规模级别)

  Phase 3b (训练验证):
    --exp5: 主动生成记忆图训练数据 (500条对话→记忆图 pairs)
    --exp6: 零样本转移测试数据 (held-out 任务类型)

用法:
  python phase3_data_gen.py --all --output ./phase3_data/
  python phase3_data_gen.py --exp1 --output ./phase3_data/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  共享模板 — 多主题对话素材
# ═══════════════════════════════════════════════════════════════

TOPIC_TEMPLATES = [
    # (话题领域, [子话题列表], 关联话题索引)
    {
        "domain": "软件架构",
        "subtopics": [
            "微服务拆分", "API 网关设计", "数据库选型", "缓存策略",
            "消息队列", "服务降级", "分布式事务", "配置中心",
        ],
        "links_to": [1, 2],
    },
    {
        "domain": "AI 应用",
        "subtopics": [
            "模型选型", "Prompt 工程", "RAG 检索增强", "Agent 架构",
            "微调策略", "推理加速", "多模态集成", "评估体系",
        ],
        "links_to": [0, 3],
    },
    {
        "domain": "团队管理",
        "subtopics": [
            "代码审查流程", "技术债务管理", "新人 onboarding", "远程协作",
            "Sprint 规划", "KPI 设定", "技术分享", "文档规范",
        ],
        "links_to": [0, 4],
    },
    {
        "domain": "产品设计",
        "subtopics": [
            "用户调研", "原型设计", "A/B 测试", "数据埋点",
            "增长策略", "竞品分析", "需求优先级", "用户体验",
        ],
        "links_to": [1, 4],
    },
    {
        "domain": "性能优化",
        "subtopics": [
            "慢查询优化", "CDN 加速", "前端打包", "内存泄漏",
            "并发模型", "索引优化", "负载均衡", "压测方案",
        ],
        "links_to": [0, 2],
    },
]

USER_COMMENTS = [
    "这个很重要，需要重点跟进",
    "后面再讨论这个",
    "我觉得这个方向不太对",
    "先记下来，回头再评估",
    "这是核心问题，必须解决",
    "暂时搁置，优先级不高",
    "这个方案不错，但实施成本可能太高",
    "需要和其他团队对齐",
    "这里有个坑，之前踩过",
    "参考一下业界最佳实践",
    "可以先做 MVP 验证",
    "这个和上周讨论的类似，可以复用",
]

AGENT_RESPONSES = [
    "好的，我记下了。这个涉及到多个模块的联动，需要全局考虑。",
    "明白，我会在后续的分析中重点关注这部分。",
    "从系统角度来看，这个改动可能会影响几个相关的服务。",
    "我理解你的顾虑。从架构层面看，这个问题确实需要谨慎处理。",
    "让我先把这个问题标记为高优先级，后续展开详细分析。",
    "基于之前的讨论，这个方向和我们的整体架构是一致的。",
    "这里有几点需要考虑：第一是兼容性，第二是扩展性，第三是维护成本。",
    "我注意到这个和上次讨论的话题有关联，建议一并考虑。",
]

# 更丰富的用户消息模板（减少模板重复感）
USER_TEMPLATES = [
    "关于 {domain} 的 {topic}，我有些想法。{comment}",
    "我想讨论一下 {topic} 的方案，之前调研过几种做法。{comment}",
    "{topic} 这块我觉得目前还有优化空间，你怎么看？{comment}",
    "最近在看 {domain} 相关的 {topic}，发现有些问题。{comment}",
    "上次提到的 {topic}，我回去想了想。{comment}",
    "对比了几个 {topic} 的实现方案，优缺点各有。{comment}",
    "关于 {topic}，团队内部有不同意见。{comment}",
    "我整理了一下 {topic} 的需求文档。{comment}",
]

AGENT_TEMPLATES = [
    "好的，{topic} 确实是个需要仔细考虑的点。从整体架构来看，它和哪些模块有耦合？",
    "了解。{topic} 这个方向我之前也关注过，有几个关键点要注意。",
    "收到。{topic} 我可以先从几个角度帮你梳理一下。",
    "明白，这个我记下来。{topic} 的优先级你觉得应该怎么定？",
    "关于 {topic}，能不能展开说说具体是哪个方面让你觉得有问题？",
    "好的，{topic} 这边我可以提供一些业界的参考案例。",
    "我理解了。{topic} 和上轮聊的有没有冲突的地方？",
    "明白。我先整理一下 {topic} 的要点，后续可以作为决策依据。",
]


def _gen_conversation(domain_idx: int, num_rounds: int = 20,
                      with_annotations: bool = True) -> tuple[list[dict], dict]:
    """生成一段合成对话。

    Returns:
        (conversation, annotations)
    """
    domain = TOPIC_TEMPLATES[domain_idx]
    subtopics = domain["subtopics"]
    conversation = []
    annotations = {}

    round_num = 1
    current_topic_idx = 0
    topic_cycle = list(range(len(subtopics)))
    random.shuffle(topic_cycle)

    while round_num <= num_rounds:
        topic_idx = topic_cycle[current_topic_idx % len(topic_cycle)]
        topic = subtopics[topic_idx]

        user_msg = {
            "round": round_num,
            "role": "user",
            "content": random.choice(USER_TEMPLATES).format(
                domain=domain["domain"], topic=topic, comment=random.choice(USER_COMMENTS)
            ),
        }
        conversation.append(user_msg)

        if with_annotations and random.random() < 0.25:
            annotations[str(round_num)] = {
                "circle": True,
                "circle_text": topic,
                "importance": "high",
                "comment": user_msg["content"][:50],
            }

        round_num += 1

        agent_msg = {
            "round": round_num,
            "role": "assistant",
            "content": random.choice(AGENT_TEMPLATES).format(topic=topic),
        }
        conversation.append(agent_msg)
        round_num += 1

        if random.random() < 0.3:
            detail_rounds = random.randint(1, 4)
            for _ in range(detail_rounds):
                if round_num > num_rounds:
                    break
                detail_user = {
                    "round": round_num,
                    "role": "user",
                    "content": f"具体来说，{topic} 方面我们需要考虑 {random.choice(['兼容性', '扩展性', '性能', '安全性', '可维护性'])}。"
                              f"比如 {random.choice(['接口设计', '数据迁移', '灰度发布', '回滚方案', '监控告警'])}这块。"
                              f"{random.choice(USER_COMMENTS)}",
                }
                conversation.append(detail_user)

                if with_annotations and random.random() < 0.2:
                    annotations[str(round_num)] = {
                        "circle": random.random() < 0.5,
                        "circle_text": f"{topic} 细节",
                        "importance": "high" if random.random() < 0.6 else "medium",
                        "comment": detail_user["content"][:60],
                    }
                round_num += 1

                if round_num > num_rounds:
                    break
                detail_agent = {
                    "round": round_num,
                    "role": "assistant",
                    "content": random.choice(AGENT_RESPONSES),
                }
                conversation.append(detail_agent)
                round_num += 1

        current_topic_idx += 1
        if current_topic_idx >= len(topic_cycle):
            random.shuffle(topic_cycle)
            current_topic_idx = 0

    return conversation, annotations


def _generate_qa_pairs(conversation: list[dict], annotations: dict,
                       memory_graph: dict) -> list[dict]:
    """为一组对话+记忆图生成 QA 对 (中性 ground truth，不含图术语)。

    问题类型:
      - fact: 事实回忆 (对话中讨论了什么)- 文本和图都可回答
      - relation: 话题关联 - 图优势但文本也能推理
      - marker: 用户标注了什么重点 - 图优势 (标注在图上有视觉标记)
      - summary: 对话整体总结 - 公平对比
    """
    qa_pairs = []
    nodes = memory_graph.get("nodes", [])
    edges = memory_graph.get("edges", [])

    if not nodes:
        return qa_pairs

    high_nodes = [n for n in nodes if n.get("importance") == "high"]

    # 事实回忆 — 文本和图都能回答
    for i in [len(conversation)//4, len(conversation)//2, len(conversation)*3//4]:
        if 0 <= i < len(conversation):
            msg = conversation[i]
            content = msg["content"]
            topic_hint = content.split("。")[0][:60]
            qa_pairs.append({
                "type": "fact",
                "question": f"对话中段讨论了什么内容？",
                "answer": topic_hint,
                "round": msg["round"],
            })

    # 话题关联 — 图可以直接看拓扑，文本需要推断
    if len(edges) >= 2:
        e1, e2 = edges[0], edges[1]
        src1 = next((n["label"] for n in nodes if n["id"] == e1["from"]), e1["from"])
        dst1 = next((n["label"] for n in nodes if n["id"] == e1["to"]), e1["to"])
        src2 = next((n["label"] for n in nodes if n["id"] == e2["from"]), e2["from"])
        dst2 = next((n["label"] for n in nodes if n["id"] == e2["to"]), e2["to"])
        qa_pairs.append({
            "type": "relation",
            "question": f"对话中提到的 {src1} 和 {dst1} 这两个话题是如何关联的？"
                        f"它们的讨论顺序和内容有什么联系？",
            "answer": f"{src1} 和 {dst1} 在对话中先后被讨论，"
                      f"两者围绕同一主题领域展开，内容上有衔接关系。",
        })

    # 标注提取 — 图独有优势（圈注/星标）
    if high_nodes:
        names = "、".join(n["label"] for n in high_nodes[:5])
        qa_pairs.append({
            "type": "marker",
            "question": "对话中哪些话题得到了最多的讨论或最受重视？",
            "answer": f"被重点讨论的话题包括: {names}，这些话题在对话中反复出现。",
        })

    # 对话总结 — 公平对比题
    domain_topics = list(set(n["label"] for n in nodes))
    topic_sample = "、".join(domain_topics[:6])
    qa_pairs.append({
        "type": "summary",
        "question": "请简要总结这段对话主要讨论了哪些方面？",
        "answer": f"对话主要围绕 {topic_sample} 等方面展开，"
                  f"涉及 {len(nodes)} 个相关话题的讨论。",
    })

    return qa_pairs[:6]



# ═══════════════════════════════════════════════════════════════
#  Exp1: 静态记忆图 — "扫图即回忆"
# ═══════════════════════════════════════════════════════════════

def gen_exp1(output_dir: str, num_samples: int = 30):
    """生成 Exp1 数据: 30段对话 → 记忆图 + QA"""
    from phase3.memory_graph import build_memory_graph

    out_dir = Path(output_dir) / "exp1_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for sample_id in range(num_samples):
        domain_idx = sample_id % len(TOPIC_TEMPLATES)
        num_rounds = random.randint(15, 35)

        conversation, annotations = _gen_conversation(domain_idx, num_rounds, with_annotations=True)
        memory_graph = build_memory_graph(conversation, annotations, version=1)
        qa_pairs = _generate_qa_pairs(conversation, annotations, memory_graph)

        sample = {
            "sample_id": f"exp1_{sample_id:03d}",
            "domain": TOPIC_TEMPLATES[domain_idx]["domain"],
            "num_rounds": num_rounds,
            "conversation": conversation,
            "annotations": annotations,
            "memory_graph": memory_graph,
            "qa_pairs": qa_pairs,
        }
        all_samples.append(sample)

    output_path = out_dir / "exp1_static_memory.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"[Exp1] 生成 {len(all_samples)} 个样本 → {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
#  Exp2: 增量记忆积累 — "越积越厚的笔记本"
# ═══════════════════════════════════════════════════════════════

def gen_exp2(output_dir: str, num_sessions: int = 5):
    """生成 Exp2 数据: 5段跨域连续对话，多版本记忆图。

    设计: 不同 session 使用不同领域，测试:
      - Session 1-4: 新领域 → 旧话题是否衰减
      - Session 5: 回到 Session 1 的领域 → 旧话题是否复活
    """
    from phase3.memory_graph import build_memory_graph

    out_dir = Path(output_dir) / "exp2_incremental"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 跨域设计: 5 个 session 使用不同的 domain
    # Session 5 回到 Session 1 的 domain 测试"复活"
    domain_plan = [0, 1, 2, 3, 0]  # indices into TOPIC_TEMPLATES
    assert len(domain_plan) == num_sessions

    all_conversations = []
    sessions = []

    for session_id in range(num_sessions):
        domain_idx = domain_plan[session_id]
        num_rounds = random.randint(15, 30)
        conv, annotations = _gen_conversation(domain_idx, num_rounds, with_annotations=True)
        all_conversations.append({"conv": conv, "annotations": annotations})
        sessions.append({
            "session_id": session_id + 1,
            "num_rounds": num_rounds,
            "conversation": conv,
            "annotations": annotations,
        })

    accumulated_conv = []
    accumulated_annotations = {}
    memory_versions = []
    last_graph = None
    global_round = 0

    for session_id, session_data in enumerate(sessions):
        conv = session_data["conversation"]
        ann = session_data["annotations"]

        offset_conv = []
        for msg in conv:
            msg = dict(msg)
            msg["round"] = msg["round"] + global_round
            offset_conv.append(msg)

        offset_ann = {}
        for k, v in ann.items():
            offset_ann[str(int(k) + global_round)] = v

        accumulated_conv.extend(offset_conv)
        accumulated_annotations.update(offset_ann)
        global_round += len(conv)

        graph = build_memory_graph(
            offset_conv,          # 仅传入新 session 的对话，累积累加交给 _merge_nodes
            offset_ann,
            existing_graph=last_graph,
            version=session_id + 1,
        )
        last_graph = graph

        memory_versions.append({
            "version": session_id + 1,
            "sessions_covered": list(range(1, session_id + 2)),
            "total_rounds": global_round,
            "memory_graph": graph,
        })

    qa_retention = []
    for v in memory_versions:
        graph = v["memory_graph"]
        old_nodes_in_newest = [
            n for n in graph.get("nodes", [])
            if n.get("time_factor", 0) < 0.4
        ]
        qa_retention.append({
            "version": v["version"],
            "total_nodes": len(graph.get("nodes", [])),
            "old_nodes_count": len(old_nodes_in_newest),
            "questions": [
                {
                    "question": f"在当前的记忆图 (v{v['version']}) 中，还有多少早期对话的话题被保留？",
                    "answer": f"有 {len(old_nodes_in_newest)} 个旧话题被保留在图记忆中",
                },
                {
                    "question": "请回顾最早对话中用户标记为重点的内容。",
                    "answer": f"最早标记的重点: {', '.join(n['label'] for n in old_nodes_in_newest[:3]) if old_nodes_in_newest else '无'}",
                },
                {
                    "question": "新对话 (最新) 和旧对话 (最早) 之间有什么联系？",
                    "answer": f"对话跨越了多个领域，"
                              f"新对话与旧话题间有 {len(_find_cross_session_edges(graph))} 条潜在关联",
                },
            ],
        })

    result = {
        "domain": "跨领域(架构→AI→管理→产品→架构)",
        "num_sessions": num_sessions,
        "sessions": sessions,
        "memory_versions": memory_versions,
        "qa_retention": qa_retention,
    }

    output_path = out_dir / "exp2_incremental_memory.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[Exp2] 生成 {num_sessions} 个 session，{len(memory_versions)} 个版本 → {output_path}")
    return str(output_path)


def _find_cross_session_edges(graph: dict) -> list:
    edges = graph.get("edges", [])
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    old_ids = {nid for nid, n in nodes.items() if n.get("time_factor", 0) < 0.4}
    new_ids = {nid for nid, n in nodes.items() if n.get("time_factor", 0) > 0.6}
    cross = [e for e in edges
             if (e["from"] in old_ids and e["to"] in new_ids) or
                (e["to"] in old_ids and e["from"] in new_ids)]
    return cross


# ═══════════════════════════════════════════════════════════════
#  Exp3: 标注效果消融 — 圈注/评语/高亮的作用
# ═══════════════════════════════════════════════════════════════

def gen_exp3(output_dir: str, num_samples: int = 20):
    """生成 Exp3 数据: 同一对话 → 4 版记忆图 (不同标注级别)"""
    from phase3.memory_graph import build_memory_graph

    out_dir = Path(output_dir) / "exp3_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for sample_id in range(num_samples):
        domain_idx = sample_id % len(TOPIC_TEMPLATES)
        num_rounds = random.randint(20, 35)

        conversation, annotations = _gen_conversation(domain_idx, num_rounds, with_annotations=True)

        full_graph = build_memory_graph(conversation, annotations, version=1)

        skeleton_graph = build_memory_graph(conversation, {}, version=1)
        for n in skeleton_graph["nodes"]:
            n["annotations"] = []
        for n in skeleton_graph["nodes"]:
            n["time_factor"] = 0.5

        time_only_graph = build_memory_graph(conversation, {}, version=1)
        for n in time_only_graph["nodes"]:
            n["annotations"] = []

        circle_only_graph = build_memory_graph(conversation, annotations, version=1)
        for n in circle_only_graph["nodes"]:
            n["annotations"] = [a for a in n.get("annotations", []) if a.get("type") != "comment"]
            n["time_factor"] = 0.5

        sample = {
            "sample_id": f"exp3_{sample_id:03d}",
            "domain": TOPIC_TEMPLATES[domain_idx]["domain"],
            "num_rounds": num_rounds,
            "conversation": conversation,
            "annotations": annotations,
            "graphs": {
                "A_skeleton": skeleton_graph,
                "B_time_color": time_only_graph,
                "C_circle_marker": circle_only_graph,
                "D_full_memory": full_graph,
            },
            "qa_pairs": _generate_qa_pairs(conversation, annotations, full_graph),
        }
        all_samples.append(sample)

    output_path = out_dir / "exp3_ablation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"[Exp3] 生成 {len(all_samples)} 个样本 x 4 版图 → {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
#  Exp4: 规模扩展 — 信息量-优势曲线
# ═══════════════════════════════════════════════════════════════

def gen_exp4(output_dir: str):
    """生成 Exp4 数据: 4 个规模级别 (20/100/300/600轮) 的对话"""
    from phase3.memory_graph import build_memory_graph

    out_dir = Path(output_dir) / "exp4_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [
        ("small", 20),
        ("medium", 100),
        ("large", 300),
        ("xlarge", 600),
    ]

    all_data = []

    for size_label, num_rounds in sizes:
        domain_idx = random.randint(0, len(TOPIC_TEMPLATES) - 1)
        conversation, annotations = _gen_conversation(domain_idx, num_rounds, with_annotations=True)
        memory_graph = build_memory_graph(conversation, annotations, version=1)

        qa_pairs = _generate_qa_pairs(conversation, annotations, memory_graph)

        size_data = {
            "size": size_label,
            "num_rounds": num_rounds,
            "num_topics": len(memory_graph.get("nodes", [])),
            "num_edges": len(memory_graph.get("edges", [])),
            "conversation": conversation,
            "annotations": annotations,
            "memory_graph": memory_graph,
            "qa_pairs": qa_pairs,
        }
        all_data.append(size_data)

        size_dir = out_dir / size_label
        size_dir.mkdir(parents=True, exist_ok=True)
        size_path = size_dir / f"conversation_{size_label}.json"
        with open(size_path, "w", encoding="utf-8") as f:
            json.dump(size_data, f, ensure_ascii=False, indent=2)

    output_path = out_dir / "exp4_scale_all.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"[Exp4] 生成 4 个规模级别 → {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
#  Exp5: 主动生成记忆图训练数据 (Phase 3b)
# ═══════════════════════════════════════════════════════════════

def gen_exp5(output_dir: str, num_samples: int = 500):
    """生成 Exp5 训练数据: 对话 → 记忆图 (Mermaid 格式)

    格式: [对话] → [<memory_sketch>Mermaid</memory_sketch> + 总结]
    用于微调模型学会主动画记忆图。
    """
    from phase3.memory_graph import build_memory_graph

    out_dir = Path(output_dir) / "exp5_train"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for sample_id in range(num_samples):
        domain_idx = sample_id % len(TOPIC_TEMPLATES)
        num_rounds = random.randint(15, 40)

        conversation, annotations = _gen_conversation(domain_idx, num_rounds, with_annotations=True)
        memory_graph = build_memory_graph(conversation, annotations, version=1)

        mermaid_code = _graph_to_mermaid(memory_graph)
        summary = _generate_summary(memory_graph, TOPIC_TEMPLATES[domain_idx]["domain"])

        conversation_text = _format_conversation_text(conversation)

        prompt = (
            f"以下是一段对话记录。请分析对话内容，先画出记忆图（Mermaid 格式）梳理话题结构，"
            f"标记重点，然后总结核心要点。\n\n"
            f"## 对话记录\n\n{conversation_text}\n\n"
            f"请输出记忆图和总结。"
        )

        response = (
            f"<memory_sketch>\n{mermaid_code}\n</memory_sketch>\n\n"
            f"<summary>\n{summary}\n</summary>"
        )

        samples.append({
            "id": f"exp5_{sample_id:04d}",
            "domain": TOPIC_TEMPLATES[domain_idx]["domain"],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "memory_graph": memory_graph,
        })

    output_path = out_dir / "exp5_train_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[Exp5] 生成 {len(samples)} 条训练样本 → {output_path}")
    return str(output_path)


def _graph_to_mermaid(graph: dict) -> str:
    """将记忆图转为 Mermaid 代码"""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    lines = ["graph TD"]

    for n in nodes:
        nid = n["id"].replace("-", "_").replace(" ", "_")
        label = n.get("label", nid)
        imp = n.get("importance", "normal")
        style = "fill:#ff6b6b,color:#fff" if imp == "high" else (
            "fill:#ffd93d" if imp == "medium" else "fill:#e9ecef"
        )
        annotations = ""
        for ann in n.get("annotations", []):
            if ann.get("type") == "circle":
                annotations += " [★重点关注]"
            elif ann.get("type") == "comment":
                annotations += f" [评: {ann.get('text', '')[:20]}]"
        lines.append(f"    {nid}[\"{label}{annotations}\"]")
        lines.append(f"    style {nid} {style}")

    for e in edges:
        src = e["from"].replace("-", "_").replace(" ", "_")
        dst = e["to"].replace("-", "_").replace(" ", "_")
        etype = e.get("type", "references")
        if etype == "causes":
            arrow = "-->"
        elif etype == "contradicts":
            arrow = "<-->"
        elif etype == "extends":
            arrow = "-.->"
        else:
            arrow = "-->"
        lines.append(f"    {src} {arrow}|{etype}| {dst}")

    return "\n".join(lines)


def _generate_summary(memory_graph: dict, domain: str) -> str:
    """生成对话总结"""
    nodes = memory_graph.get("nodes", [])
    edges = memory_graph.get("edges", [])

    high_nodes = [n for n in nodes if n.get("importance") == "high"]
    high_names = ", ".join(n["label"] for n in high_nodes[:5]) if high_nodes else "无"

    summary_parts = [
        f"## {domain} 对话总结",
        f"",
        f"本次对话共涉及 {len(nodes)} 个话题，其中重点关注的话题有: {high_names}。",
        f"话题之间共建立了 {len(edges)} 条关联关系。",
    ]

    annotated = [n for n in nodes if n.get("annotations")]
    if annotated:
        summary_parts.append(f"\n**用户标注要点:**")
        for n in annotated[:5]:
            for ann in n.get("annotations", []):
                if ann.get("type") == "circle":
                    summary_parts.append(f"- ★ {n['label']}: {ann.get('text', '')}")
                elif ann.get("type") == "comment":
                    summary_parts.append(f"- {n['label']}: {ann.get('text', '')}")

    return "\n".join(summary_parts)


def _format_conversation_text(conversation: list[dict]) -> str:
    """格式化对话为文本"""
    lines = []
    for msg in conversation:
        role = "用户" if msg.get("role") == "user" else "助手"
        r = msg.get("round", "?")
        lines.append(f"[第{r}轮] {role}: {msg.get('content', '')}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Exp6: 零样本转移测试数据 (Phase 3b, 可选)
# ═══════════════════════════════════════════════════════════════

def gen_exp6(output_dir: str, num_samples: int = 50):
    """生成 Exp6 数据: held-out 任务类型的零样本转移测试"""
    out_dir = Path(output_dir) / "exp6_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    held_out_tasks = [
        {
            "task": "meeting_minutes",
            "name": "会议纪要分析",
            "prompt_template": "以下是会议纪要。请分析会议讨论的主要议题和关联关系。\n\n{text}",
        },
        {
            "task": "research_survey",
            "name": "文献综述整理",
            "prompt_template": "以下是多篇文献摘要。请梳理研究主题之间的关系。\n\n{text}",
        },
        {
            "task": "customer_support",
            "name": "客服工单分析",
            "prompt_template": "以下是客户支持对话记录。请识别核心问题和解决方案路径。\n\n{text}",
        },
        {
            "task": "code_review",
            "name": "代码审查总结",
            "prompt_template": "以下是代码审查评论。请整理审查要点和修改建议的关系。\n\n{text}",
        },
        {
            "task": "debate_transcript",
            "name": "辩论记录分析",
            "prompt_template": "以下是辩论记录。请梳理正反方论点和论证链。\n\n{text}",
        },
    ]

    samples = []
    for task_info in held_out_tasks:
        for i in range(num_samples // len(held_out_tasks)):
            domain_idx = random.randint(0, len(TOPIC_TEMPLATES) - 1)
            conv, annotations = _gen_conversation(domain_idx, 30, with_annotations=False)
            text = _format_conversation_text(conv)
            samples.append({
                "id": f"exp6_{task_info['task']}_{i:03d}",
                "task": task_info["task"],
                "task_name": task_info["name"],
                "prompt": task_info["prompt_template"].format(text=text),
                "conversation": conv,
                "expects_graph_behavior": True,
            })

    output_path = out_dir / "exp6_transfer.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[Exp6] 生成 {len(samples)} 条零样本转移测试 → {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 实验数据生成器")
    parser.add_argument("--all", action="store_true", help="生成全部实验数据")
    parser.add_argument("--exp1", action="store_true", help="Exp1: 静态记忆图")
    parser.add_argument("--exp2", action="store_true", help="Exp2: 增量记忆积累")
    parser.add_argument("--exp3", action="store_true", help="Exp3: 标注效果消融")
    parser.add_argument("--exp4", action="store_true", help="Exp4: 规模扩展")
    parser.add_argument("--exp5", action="store_true", help="Exp5: 主动生成训练数据")
    parser.add_argument("--exp6", action="store_true", help="Exp6: 零样本转移测试")
    parser.add_argument("--output", "-o", default="./phase3_data/", help="输出目录")
    parser.add_argument("--num-samples", type=int, default=None, help="覆盖默认样本数")
    args = parser.parse_args()

    run_all = args.all

    if run_all or args.exp1:
        gen_exp1(args.output)
    if run_all or args.exp2:
        gen_exp2(args.output)
    if run_all or args.exp3:
        gen_exp3(args.output)
    if run_all or args.exp4:
        gen_exp4(args.output)
    if run_all or args.exp5:
        ns = args.num_samples or 500
        gen_exp5(args.output, ns)
    if run_all or args.exp6:
        ns = args.num_samples or 50
        gen_exp6(args.output, ns)

    if not any([run_all, args.exp1, args.exp2, args.exp3, args.exp4, args.exp5, args.exp6]):
        parser.print_help()
        print("\n提示: 使用 --all 生成全部实验数据")


if __name__ == "__main__":
    main()
