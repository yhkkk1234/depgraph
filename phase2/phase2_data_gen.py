#!/usr/bin/env python
"""
phase2_data_gen — Phase 2 实验数据生成器

生成 4 个实验所需的训练/测试数据:
  --exp1: 主动画图训练数据 (500 条多域样本)
  --exp2: 长对话 + 记忆测试数据 (50/100/150/200轮)
  --exp3: 图复杂度测试数据 (3 级别)
  --exp4: 零样本迁移测试数据 (3 任务类型)
  --all:  生成全部

用法:
  python phase2_data_gen.py --all --output ./training_data/
  python phase2_data_gen.py --exp2 --rounds 50,100,200 --output ./test_data/
  python phase2_data_gen.py --exp3 --complexity simple,moderate,complex
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

# 固定随机种子以保证可复现
random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  共享模板数据
# ═══════════════════════════════════════════════════════════════

MULTI_FILE_CODE_TASKS = [
    {
        "scenario": "重构支付模块的退款逻辑",
        "modules": ["payment/refund.py", "order/status.py", "accounting/ledger.py", "notification/email.py", "api/refund_handler.py"],
        "change_module": "payment/refund.py",
        "affected": ["order/status.py", "accounting/ledger.py", "notification/email.py", "api/refund_handler.py"],
        "bug_desc": "修改 refund.process() 的参数从 (order_id, amount) 改为 (order_id, amount, reason, operator_id)。所有调用方需要同步更新。",
    },
    {
        "scenario": "添加用户认证中间件",
        "modules": ["auth/middleware.py", "auth/token.py", "api/v1/users.py", "api/v1/products.py", "api/v1/orders.py", "utils/cache.py"],
        "change_module": "auth/middleware.py",
        "affected": ["api/v1/users.py", "api/v1/products.py", "api/v1/orders.py", "auth/token.py", "utils/cache.py"],
        "bug_desc": "在 API 层统一添加认证中间件调用。所有 API 路由需要在 handler 前调用 auth.validate_token()。",
    },
    {
        "scenario": "数据库迁移：用户表结构调整",
        "modules": ["models/user.py", "services/user_service.py", "services/auth_service.py", "admin/user_admin.py"],
        "change_module": "models/user.py",
        "affected": ["services/user_service.py", "services/auth_service.py", "admin/user_admin.py"],
        "bug_desc": "User 模型新增 email_verified 字段和 last_login_ip 字段。所有使用 User 对象的服务需要适配。",
    },
    # ── v2 新增模板 (提升多样性) ──
    {
        "scenario": "配置系统重构",
        "modules": ["config/settings.py", "config/loader.py", "services/app_service.py", "services/cache_service.py", "cli/main.py"],
        "change_module": "config/settings.py",
        "affected": ["config/loader.py", "services/app_service.py", "services/cache_service.py", "cli/main.py"],
        "bug_desc": "将 Settings.load() 改为异步方法 async def load()。所有调用 load() 的地方需要添加 await。",
    },
    {
        "scenario": "日志模块接口变更",
        "modules": ["utils/logger.py", "services/webhook.py", "services/monitor.py", "api/middleware.py", "tasks/cleanup.py"],
        "change_module": "utils/logger.py",
        "affected": ["services/webhook.py", "services/monitor.py", "api/middleware.py", "tasks/cleanup.py"],
        "bug_desc": "Logger.log() 新增 level 参数 (默认为 INFO)。需要检查所有调用点是否指定了正确的日志级别。",
    },
    {
        "scenario": "消息队列迁移",
        "modules": ["mq/broker.py", "mq/consumer.py", "services/order_handler.py", "services/notification.py", "tasks/retry.py"],
        "change_module": "mq/broker.py",
        "affected": ["mq/consumer.py", "services/order_handler.py", "services/notification.py", "tasks/retry.py"],
        "bug_desc": "Broker.publish() 返回类型从 bool 改为 MessageID 对象。所有调用方需要适配新的返回值类型。",
    },
    {
        "scenario": "缓存接口统一",
        "modules": ["cache/redis_client.py", "cache/local_cache.py", "services/session.py", "services/rate_limiter.py", "api/handlers.py"],
        "change_module": "cache/redis_client.py",
        "affected": ["cache/local_cache.py", "services/session.py", "services/rate_limiter.py", "api/handlers.py"],
        "bug_desc": "RedisClient 新增 TTL 参数，get()/set() 签名变更。所有缓存使用方需要传递 TTL 或使用默认值。",
    },
    {
        "scenario": "文件存储抽象层",
        "modules": ["storage/base.py", "storage/s3.py", "storage/local.py", "services/upload.py", "admin/media.py"],
        "change_module": "storage/base.py",
        "affected": ["storage/s3.py", "storage/local.py", "services/upload.py", "admin/media.py"],
        "bug_desc": "BaseStorage.save() 新增 metadata 字典参数。所有存储后端和调用方需要更新。",
    },
]

WRITING_TASKS = [
    {
        "scenario": "科技评论文章结构分析",
        "text": """本文讨论自动驾驶的安全问题。作者提出三个论点：(1) 统计数据表明自动驾驶事故率低于人类驾驶；(2) 但公众对机器失误的容忍度远低于人类失误；(3) 因此需要建立新的安全认证体系，但这可能延缓技术推广。论点之间存在矛盾：效率 vs 审慎。""",
        "expected_topics": ["自动驾驶安全", "统计数据", "公众容忍度", "安全认证", "技术推广"],
        "expected_relations": [
            {"from": "自动驾驶安全", "to": "统计数据", "type": "references"},
            {"from": "自动驾驶安全", "to": "公众容忍度", "type": "contradicts"},
            {"from": "公众容忍度", "to": "安全认证", "type": "causes"},
            {"from": "安全认证", "to": "技术推广", "type": "contradicts"},
        ],
    },
    {
        "scenario": "政策建议报告评估",
        "text": """关于城市垃圾分类政策的建议：主张强制分类（论点A），但有实施成本问题（论点B）；替代方案是激励性分类（论点C），但效果存疑（论点D）；折中方案是试点先行（论点E）。""",
        "expected_topics": ["强制分类", "实施成本", "激励分类", "效果存疑", "试点方案"],
        "expected_relations": [
            {"from": "强制分类", "to": "实施成本", "type": "contradicts"},
            {"from": "强制分类", "to": "激励分类", "type": "references"},
            {"from": "激励分类", "to": "效果存疑", "type": "contradicts"},
            {"from": "实施成本", "to": "试点方案", "type": "causes"},
        ],
    },
    # ── v2 新增模板 ──
    {
        "scenario": "开源软件商业模式分析",
        "text": """讨论开源软件的商业模式选择。核心论点：(1) 开源能快速建立用户基础；(2) 但免费模式难以支撑研发投入；(3) 双许可证模式(社区版+企业版)可兼顾两者；(4) 但可能引起社区分裂。矛盾在于：开放 vs 盈利，社区 vs 商业。""",
        "expected_topics": ["开源社区", "商业模式", "双许可证", "社区分裂", "研发投入"],
        "expected_relations": [
            {"from": "开源社区", "to": "商业模式", "type": "contradicts"},
            {"from": "开源社区", "to": "双许可证", "type": "causes"},
            {"from": "双许可证", "to": "社区分裂", "type": "contradicts"},
            {"from": "商业模式", "to": "研发投入", "type": "references"},
        ],
    },
    {
        "scenario": "远程办公优劣分析",
        "text": """远程办公的趋势分析。支持观点：(1) 提升员工满意度；(2) 减少通勤成本；(3) 扩大招聘地理范围。反对观点：(4) 团队协作效率下降；(5) 企业文化难以传承；(6) 管理难度增加。核心矛盾：灵活性 vs 协作效率。""",
        "expected_topics": ["远程办公", "员工满意度", "协作效率", "企业文化", "管理难度"],
        "expected_relations": [
            {"from": "远程办公", "to": "员工满意度", "type": "references"},
            {"from": "远程办公", "to": "协作效率", "type": "contradicts"},
            {"from": "协作效率", "to": "管理难度", "type": "causes"},
            {"from": "管理难度", "to": "企业文化", "type": "references"},
        ],
    },
    {
        "scenario": "教育技术伦理分析",
        "text": """AI 辅助教育的伦理考量。论点链：(1) AI 个性化教学能缩小教育差距；(2) 但数据采集涉及学生隐私；(3) 算法偏见可能固化教育资源不平等；(4) 需要建立 AI 教育伦理审查框架。张力：效率提升 vs 公平保障。""",
        "expected_topics": ["个性化教学", "隐私问题", "算法偏见", "伦理审查", "教育公平"],
        "expected_relations": [
            {"from": "个性化教学", "to": "隐私问题", "type": "contradicts"},
            {"from": "隐私问题", "to": "算法偏见", "type": "causes"},
            {"from": "算法偏见", "to": "伦理审查", "type": "causes"},
            {"from": "个性化教学", "to": "教育公平", "type": "contradicts"},
        ],
    },
]

CONVERSATION_TEMPLATES = {
    "code_review": {
        "topic_seeds": ["代码重构", "性能优化", "测试策略", "部署方案", "代码规范"],
        "user_styles": ["提问型", "质疑型", "补充型", "总结型"],
    },
    "project_planning": {
        "topic_seeds": ["需求分析", "架构设计", "技术选型", "风险评估", "排期规划", "分工协调"],
        "user_styles": ["发起型", "响应型", "深入型", "跳转型"],
    },
    "debate": {
        "topic_seeds": ["论点A", "反驳A", "论点B", "证据", "妥协方案", "新视角"],
        "user_styles": ["陈述型", "追问型", "举例型", "切换型"],
    },
}


# ═══════════════════════════════════════════════════════════════
#  Exp1 数据生成: 主动画图训练数据
# ═══════════════════════════════════════════════════════════════

def generate_exp1_data(output_dir: str = "./training_data/exp1") -> list[dict]:
    """生成主动画图训练数据 (v2: 800条, 多域+多模式)。"""
    print("生成 Exp1 训练数据 (v2: 800条)...", file=sys.stderr)
    samples = []
    sample_id = 0

    # ── 代码修改类 (300条, ↑100) ──
    for _ in range(300):
        task = random.choice(MULTI_FILE_CODE_TASKS)
        modules_str = ", ".join(task["modules"])
        affected_str = ", ".join(task["affected"])

        # 生成 Mermaid 图
        mermaid = "```mermaid\ngraph TD\n"
        mermaid += f"    {task['change_module'].replace('/', '_').replace('.', '_')}"
        mermaid += f"[\"{task['change_module']}<br/>CHANGED\"]\n"
        mermaid += "    style " + task['change_module'].replace('/', '_').replace('.', '_') + " fill:#ff4444\n"
        for aff in task["affected"]:
            node_id = aff.replace('/', '_').replace('.', '_')
            mermaid += f"    {node_id}[\"{aff}\"]\n"
            mermaid += f"    style {node_id} fill:#ffd43b\n"
        mermaid += "```"

        assistant = f"""<sketch>
{mermaid}
</sketch>

<analysis>
修改模块: {task['change_module']}
直接影响文件: {affected_str}

影响分析:
1. {task['change_module']} 的接口变更会影响所有调用方
2. 需要检查每个受影响的文件中的具体调用点
3. 重点验证: 参数传递是否正确、类型是否匹配
</analysis>

我将逐一修改以下文件:

{chr(10).join(f"- {f}" for f in task['affected'])}
"""

        samples.append({
            "id": f"code_{sample_id}",
            "task_type": "code_modification",
            "system_prompt": "你是一个会主动绘制依赖图来辅助分析的编程助手。在修改代码前，先用 Mermaid 画出影响关系图。",
            "user": f"项目模块: {modules_str}\n修改描述: {task['bug_desc']}\n当前修改模块: {task['change_module']}",
            "assistant": assistant,
        })
        sample_id += 1

    # ── 文章分析类 (250条, ↑100) ──
    for _ in range(250):
        task = random.choice(WRITING_TASKS)
        topics_str = ", ".join(task["expected_topics"])

        mermaid = "```mermaid\ngraph TD\n"
        for t in task["expected_topics"]:
            node_id = t.replace(" ", "_")
            mermaid += f"    {node_id}[\"{t}\"]\n"
        for rel in task["expected_relations"]:
            from_id = rel["from"].replace(" ", "_")
            to_id = rel["to"].replace(" ", "_")
            mermaid += f"    {from_id} --> {to_id}\n"
        mermaid += "```"

        relations_desc = "\n".join(
            f"- {r['from']} {r['type']} {r['to']}" for r in task["expected_relations"]
        )

        assistant = f"""<sketch>
{mermaid}
</sketch>

<analysis>
话题结构:
{topics_str}

关系链:
{relations_desc}

关键发现:
- 存在矛盾关系需要在论证中调和
- 因果关系链需要验证逻辑严密性
</analysis>
"""

        samples.append({
            "id": f"writing_{sample_id}",
            "task_type": "article_outline",
            "system_prompt": "你是一个会主动绘制话题关系图来辅助分析的写作助手。",
            "user": f"请分析以下文章的结构:\n{task['text']}",
            "assistant": assistant,
        })
        sample_id += 1

    # ── 对话总结类 (250条, ↑100) ──
    for _ in range(250):
        conv_type = random.choice(list(CONVERSATION_TEMPLATES.keys()))
        template = CONVERSATION_TEMPLATES[conv_type]
        topics = template["topic_seeds"]

        mermaid = "```mermaid\ngraph TD\n"
        for t in topics:
            node_id = t.replace(" ", "_")
            mermaid += f"    {node_id}[\"{t}\"]\n"
        for i in range(len(topics) - 1):
            from_id = topics[i].replace(" ", "_")
            to_id = topics[i + 1].replace(" ", "_")
            mermaid += f"    {from_id} --> {to_id}\n"
        mermaid += "```"

        assistant = f"""<sketch>
{mermaid}
</sketch>

<analysis>
话题演变:
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(topics))}

话题转换模式: {' → '.join(topics)}
对话风格: {', '.join(template['user_styles'])}
</analysis>
"""

        samples.append({
            "id": f"conv_{sample_id}",
            "task_type": "conversation_summary",
            "system_prompt": "你是一个会主动绘制话题演化图来辅助分析的对话助手。",
            "user": f"请总结以下对话的话题演变:\n[此处插入 {conv_type} 类型对话]",
            "assistant": assistant,
        })
        sample_id += 1

    # 保存
    _save_jsonl(samples, output_dir, "active_drawing_train.jsonl")
    print(f"  生成 {len(samples)} 条训练样本", file=sys.stderr)
    return samples


# ═══════════════════════════════════════════════════════════════
#  Exp2 数据生成: 长对话 + 记忆检查点
# ═══════════════════════════════════════════════════════════════

def generate_exp2_data(output_dir: str = "./test_data/exp2",
                       round_levels: list[int] = None):
    """生成长对话测试数据，含记忆检查点和检索问题。"""
    round_levels = round_levels or [50, 100, 200]
    print("生成 Exp2 测试数据...", file=sys.stderr)

    for n_rounds in round_levels:
        print(f"  生成 {n_rounds} 轮对话...", file=sys.stderr)

        conversation = []
        memory_checkpoints = []
        topics = _pick_topics(n_rounds)

        # 生成话题生命周期
        topic_lifecycle = {}
        for t in topics:
            start = random.randint(0, max(0, n_rounds - 30))
            end = min(start + random.randint(10, 40), n_rounds)
            topic_lifecycle[t] = {"start": start, "end": end, "peaks": []}
            for _ in range(random.randint(1, 3)):
                topic_lifecycle[t]["peaks"].append(random.randint(start, end))

        for r in range(1, n_rounds + 1):
            active_topics = [t for t, lc in topic_lifecycle.items()
                           if lc["start"] <= r <= lc["end"]]

            if not active_topics:
                active_topics = [random.choice(topics)]

            main_topic = random.choice(active_topics)
            role = "user" if r % 2 == 1 else "assistant"

            msg = _generate_round_content(r, main_topic, topic_lifecycle, role)

            # 记忆检查点: 每10轮设置一个重要信息
            if r % 10 == 0 and r >= 10:
                checkpoint = {
                    "round": r,
                    "topic": main_topic,
                    "content": f"关键决策: 在{main_topic}方面，确定了{_random_decision()}",
                    "importance": "high",
                }
                memory_checkpoints.append(checkpoint)
                msg["content"] += f"\n[重要] {checkpoint['content']}"

            conversation.append(msg)

        # 生成检索问题
        fact_questions = _generate_fact_questions(conversation, memory_checkpoints, n_rounds)
        relation_questions = _generate_relation_questions(topic_lifecycle, topics, n_rounds)

        # 保存
        data = {
            "rounds": n_rounds,
            "conversation": conversation,
            "memory_checkpoints": memory_checkpoints,
            "questions": fact_questions + relation_questions,
            "topics": topics,
            "topic_lifecycle": topic_lifecycle,
        }

        out_path = Path(output_dir) / f"conversation_{n_rounds}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"    保存: {out_path} | 检查点: {len(memory_checkpoints)} | 问题: {len(fact_questions)+len(relation_questions)}",
              file=sys.stderr)


def _pick_topics(n_rounds: int) -> list[str]:
    """根据对话长度选择适当数量的话题。"""
    all_topics = [
        "技术架构", "性能优化", "安全策略", "测试方案",
        "部署流程", "团队协作", "需求变更", "技术债务",
        "监控告警", "数据管理", "用户体验", "成本控制",
    ]
    # 50轮→5话题, 100轮→7话题, 200轮→10话题
    n_topics = min(len(all_topics), max(5, n_rounds // 20))
    return random.sample(all_topics, n_topics)


def _generate_round_content(round_num: int, topic: str,
                            lifecycle: dict, role: str) -> dict:
    """生成单轮对话内容。"""
    templates = {
        "user": [
            f"关于{topic}，我有个疑问...",
            f"{topic}方面做得怎么样了？",
            f"我重新考虑了{topic}的方案，觉得可以改进",
            f"{topic}这里遇到了一个问题，能帮我分析吗？",
            f"上次说的{topic}，我想补充几点",
        ],
        "assistant": [
            f"关于{topic}，我的分析是...",
            f"{topic}方面建议采用以下方案...",
            f"针对{topic}的问题，有几个解决思路",
            f"{topic}的进展汇报：已完成核心模块",
            f"补充一下{topic}相关的内容",
        ],
    }
    content = random.choice(templates.get(role, templates["user"]))
    return {
        "round": round_num,
        "role": role,
        "topic": topic,
        "content": content,
    }


def _random_decision() -> str:
    """随机决策文本。"""
    decisions = [
        "使用 Redis 作为缓存层",
        "采用微服务架构拆分",
        "统一使用 gRPC 通信协议",
        "引入消息队列解耦",
        "切换到 PostgreSQL 数据库",
        "采用 Kubernetes 部署",
        "使用 JWT 进行身份认证",
        "引入 ELK 日志系统",
    ]
    return random.choice(decisions)


def _generate_fact_questions(conversation: list, checkpoints: list,
                             n_rounds: int) -> list[dict]:
    """生成事实性问题。"""
    questions = []
    # 从检查点中选取
    for cp in checkpoints[:5]:  # 最多5个事实问题
        questions.append({
            "type": "fact",
            "question": f"在第{cp['round']}轮中，关于{cp['topic']}做了什么关键决策？",
            "answer": [cp["content"].split(": ")[-1]] if ": " in cp["content"] else [cp["content"]],
            "round": cp["round"],
        })

    # 跨轮次引用问题
    if len(checkpoints) >= 2:
        cp1, cp2 = checkpoints[1], checkpoints[-2]
        questions.append({
            "type": "fact",
            "question": f"从第{cp1['round']}轮到第{cp2['round']}轮之间，话题{topic_key_from_cp(cp1)}和{topic_key_from_cp(cp2)}分别有哪些变化？",
            "answer": [f"{cp1['topic']}: {cp1['content']}", f"{cp2['topic']}: {cp2['content']}"],
            "cross_round": True,
        })

    return questions


def topic_key_from_cp(cp: dict) -> str:
    return cp.get("topic", "未知")


def _generate_relation_questions(lifecycle: dict, topics: list,
                                 n_rounds: int) -> list[dict]:
    """生成关系性问题 (v3: 加入图友好的结构性问题)。"""
    questions = []
    active = [t for t in topics if lifecycle[t]["end"] > lifecycle[t]["start"]]

    if len(active) >= 2:
        t1, t2 = active[0], active[1]
        questions.append({
            "type": "relation",
            "question": f"话题'{t1}'和'{t2}'在对话中是否有关联？如果有，是什么样的关系？",
            "answer": [t1, t2],
            "topics": [t1, t2],
        })

    if len(active) >= 3:
        questions.append({
            "type": "relation",
            "question": f"对话中话题的演变顺序是什么？从'{active[0]}'到'{active[-1]}'经历了哪些中间话题？",
            "answer": active[1:-1],
            "topics": active[:3],
        })

    # ── v3: 图友好的结构性问题 ──
    if len(topics) >= 3:
        # 最长话题
        longest = max(topics, key=lambda t: lifecycle[t]["end"] - lifecycle[t]["start"])
        questions.append({
            "type": "structure",
            "question": "对话中持续最久的话题是什么？（看话题关系图回答）",
            "answer": [longest],
            "graph_friendly": True,
        })

    if len(topics) >= 4:
        # 无交集话题
        no_overlap = []
        for i, t1 in enumerate(topics):
            for t2 in topics[i + 1:]:
                r1s, r1e = lifecycle[t1]["start"], lifecycle[t1]["end"]
                r2s, r2e = lifecycle[t2]["start"], lifecycle[t2]["end"]
                if r1e < r2s or r2e < r1s:
                    no_overlap.append(f"{t1}-{t2}")
        if no_overlap:
            questions.append({
                "type": "structure",
                "question": "哪些话题在对话时间线上没有交集？（看关系图回答）",
                "answer": no_overlap[:3],
                "graph_friendly": True,
            })

    return questions


# ═══════════════════════════════════════════════════════════════
#  Exp3 数据生成: 图复杂度测试数据
# ═══════════════════════════════════════════════════════════════

def generate_exp3_data(output_dir: str = "./test_data/exp3",
                       levels: list[str] = None):
    """生成不同复杂度级别的图测试数据。"""
    levels = levels or ["simple", "moderate", "complex"]
    print("生成 Exp3 测试数据...", file=sys.stderr)

    from render_diagram import render_dependency_graph
    import networkx as nx

    configs = {
        "simple": {"nodes": 6, "edges": 8, "has_cycle": False, "has_isolated": True},
        "moderate": {"nodes": 15, "edges": 24, "has_cycle": True, "has_isolated": False},
        "complex": {"nodes": 30, "edges": 55, "has_cycle": True, "has_isolated": True},
    }

    for level in levels:
        cfg = configs.get(level, configs["moderate"])
        print(f"  生成 {level} 级别图 ({cfg['nodes']}节点/{cfg['edges']}边)...", file=sys.stderr)

        # 生成随机图
        G = nx.gnm_random_graph(cfg["nodes"], cfg["edges"], seed=hash(level) % 100, directed=True)

        # 确保指定的特性
        if cfg["has_cycle"]:
            # 确保存在回路
            if not list(nx.simple_cycles(G)):
                G.add_edge(0, 1)
                G.add_edge(1, 2)
                G.add_edge(2, 0)

        # 构建 graph dict
        modules = {}
        labels = _generate_node_labels(cfg["nodes"], level)
        center_node = _find_center_node(G)

        for i in range(cfg["nodes"]):
            node_name = labels[i]
            deps = [labels[j] for j in G.successors(i) if j < cfg["nodes"]]
            modules[node_name] = {"dependencies": deps, "dependents": []}

        # 计算反向依赖
        for name in modules:
            for other_name, other_info in modules.items():
                if name in other_info.get("dependencies", []):
                    if other_name not in modules[name]["dependents"]:
                        modules[name]["dependents"].append(other_name)

        graph = {"modules": modules}

        # 计算中心节点（度数最高）
        max_deg = 0
        center_label = labels[0]
        for i in range(cfg["nodes"]):
            deg = G.in_degree(i) + G.out_degree(i)
            if deg > max_deg:
                max_deg = deg
                center_label = labels[i]

        # 检测孤立节点
        isolated = []
        for i in range(cfg["nodes"]):
            if G.degree(i) == 0:
                isolated.append(labels[i])

        # 渲染图片
        png_path = Path(output_dir) / f"graph_{level}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        render_dependency_graph(graph, output_path=str(png_path))

        # 文本描述
        text_desc = _build_text_description(cfg["nodes"], cfg["edges"],
                                            center_label, cfg["has_cycle"],
                                            isolated if cfg["has_isolated"] else [])

        # 问题集
        tasks = [
            {"type": "count", "question": "有多少个节点？", "answer": str(cfg["nodes"])},
            {"type": "center", "question": "哪个节点连接最多？", "answer": center_label},
            {"type": "cycle", "question": "是否存在回路？", "answer": "是" if cfg["has_cycle"] else "否"},
            {"type": "isolated", "question": "是否存在孤立节点？", "answer": "是" if cfg["has_isolated"] else "否"},
        ]

        data = {
            "complexity": level,
            "node_count": cfg["nodes"],
            "edge_count": cfg["edges"],
            "center_node": center_label,
            "has_cycle": cfg["has_cycle"],
            "has_isolated": cfg["has_isolated"],
            "isolated_nodes": isolated,
            "text_description": text_desc,
            "tasks": tasks,
            "ground_truth": {
                "node_count": cfg["nodes"],
                "edge_count": cfg["edges"],
                "center_node": center_label,
                "has_cycle": cfg["has_cycle"],
                "has_isolated": cfg["has_isolated"],
                "isolated_nodes": isolated,
            },
        }

        json_path = Path(output_dir) / f"graph_{level}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"    保存: {json_path} | {png_path}", file=sys.stderr)


def _generate_node_labels(n: int, prefix: str) -> list[str]:
    """生成节点标签。"""
    modules = [
        "用户服务", "订单系统", "支付网关", "库存管理", "通知中心",
        "认证模块", "日志系统", "缓存层", "数据库", "搜索引擎",
        "消息队列", "负载均衡", "API网关", "配置中心", "监控系统",
        "数据分析", "报表服务", "任务调度", "文件存储", "CDN加速",
        "推荐引擎", "搜索服务", "内容管理", "审核系统", "风控模块",
        "用户画像", "活动平台", "社交模块", "IM服务", "视频处理",
    ]
    result = modules[:n]
    while len(result) < n:
        result.append(f"{prefix}_模块{len(result)+1}")
    return result


def _find_center_node(G) -> str:
    """找度数最高的节点索引。"""
    max_deg = 0
    max_node = 0
    for node in G.nodes():
        deg = G.degree(node)
        if deg > max_deg:
            max_deg = deg
            max_node = node
    return max_node


def _build_text_description(n_nodes: int, n_edges: int, center: str,
                            has_cycle: bool, isolated: list[str]) -> str:
    """生成图的文本描述。"""
    desc = f"图中有 {n_nodes} 个节点和 {n_edges} 条有向边。"
    desc += f" 中心节点是 {center}，它的连接数最多。"
    if has_cycle:
        desc += " 图中存在至少一个回路/循环。"
    else:
        desc += " 图中没有回路，是一个有向无环图。"
    if isolated:
        desc += f" 存在 {len(isolated)} 个孤立节点: {', '.join(isolated)}。"
    else:
        desc += " 没有孤立节点，所有节点至少有一条边。"
    return desc


# ═══════════════════════════════════════════════════════════════
#  Exp4 数据生成: 零样本迁移测试
# ═══════════════════════════════════════════════════════════════

def generate_exp4_data(output_dir: str = "./test_data/exp4"):
    """生成零样本迁移测试数据。"""
    print("生成 Exp4 测试数据...", file=sys.stderr)

    tasks = {
        "decision_analysis": {
            "prompt": """你是一个决策分析助手。请分析以下多方利益相关者决策场景。

场景: 一家科技公司需要决定是否为新产品采用开源核心 + 商业插件的商业模式。

利益相关方:
- 管理层: 期望开源扩大用户基数，但担心收入
- 开发者社区: 欢迎开源但反感过度商业化
- 竞争对手: 可能利用开源代码快速跟进
- 客户: 关心产品稳定性和长期支持
- 投资人: 要求明确的盈利路径

请给出分析。""",
            "evaluation": {
                "expected_stakeholders": [
                    "管理层", "开发者社区", "竞争对手", "客户", "投资人"
                ],
                "expected_relationships": [
                    "管理层 vs 开发者社区: 商业化 vs 开源理念",
                    "开源 vs 收入: 核心矛盾",
                    "客户 vs 竞争对手: 稳定性 vs 竞争速度",
                ],
                "optimal_approach": "折中方案: 核心开源 + 企业版订阅",
            },
        },
        "argument_mapping": {
            "prompt": """分析以下辩论中的论点和论证链条。

辩题: 是否应该允许 AI 生成的代码进入生产环境？

正方 (支持):
- AI 代码生成效率是人工的 5-10 倍
- 自动化测试可以覆盖 AI 代码的质量问题
- 硅谷多家企业已在生产中使用 AI 代码

反方 (反对):
- AI 生成的代码可能包含训练数据的版权问题
- 当前 AI 缺乏对业务逻辑的深层理解
- 安全审计难以追溯 AI 代码的决策过程
- AI 代码的维护成本可能高于编写成本

请分析双方的论证结构和逻辑链条。""",
            "evaluation": {
                "expected_arguments": [
                    "效率优势", "测试覆盖", "业界实践",
                    "版权风险", "理解局限", "审计困难", "维护成本",
                ],
                "expected_structure": "逐点反驳模式",
            },
        },
        "knowledge_graph": {
            "prompt": """基于以下事实构建关系并回答查询。

事实:
- Alice 是工程副总裁
- Bob 向 Alice 汇报
- Bob 领导支付团队
- 支付团队负责 PaymentService
- PaymentService 调用了 NotificationService
- NotificationService 由 Charlie 团队负责
- Charlie 向 Alice 汇报
- Diana 是 Charlie 团队的工程师
- 最近的支付故障影响到了 NotificationService

问题:
1. PaymentService 的故障会影响哪些人？
2. Alice 需要找谁了解故障根因？
3. Diana 的工作与支付故障有什么间接关联？""",
            "evaluation": {
                "expected_entities": ["Alice", "Bob", "Charlie", "Diana"],
                "expected_paths": [
                    "PaymentService → NotificationService → Charlie团队 → Diana",
                    "Bob → Alice ← Charlie",
                ],
                "answer_patterns": ["间接关联", "汇报链", "服务依赖"],
            },
        },
    }

    out_path = Path(output_dir) / "exp4_tasks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"  保存: {out_path} | 任务: {len(tasks)} 个", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════

def _save_jsonl(samples: list[dict], output_dir: str, filename: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 实验数据生成器")
    parser.add_argument("--all", action="store_true", help="生成所有实验数据")
    parser.add_argument("--exp1", action="store_true", help="生成 Exp1 训练数据")
    parser.add_argument("--exp2", action="store_true", help="生成 Exp2 长对话数据")
    parser.add_argument("--exp3", action="store_true", help="生成 Exp3 图复杂度数据")
    parser.add_argument("--exp4", action="store_true", help="生成 Exp4 迁移测试数据")
    parser.add_argument("--output", "-o", default="./test_data/",
                        help="输出目录 (默认: ./test_data/)")
    parser.add_argument("--rounds", "-r", default="50,100,200",
                        help="Exp2: 对话轮数 (逗号分隔)")
    parser.add_argument("--complexity", "-c", default="simple,moderate,complex",
                        help="Exp3: 复杂度级别")
    args = parser.parse_args()

    if not any([args.all, args.exp1, args.exp2, args.exp3, args.exp4]):
        parser.print_help()
        return

    base_dir = Path(args.output)

    if args.all or args.exp1:
        generate_exp1_data(str(base_dir / "exp1"))

    if args.all or args.exp2:
        rounds = [int(r.strip()) for r in args.rounds.split(",")]
        generate_exp2_data(str(base_dir / "exp2"), rounds)

    if args.all or args.exp3:
        levels = [l.strip() for l in args.complexity.split(",")]
        generate_exp3_data(str(base_dir / "exp3"), levels)

    if args.all or args.exp4:
        generate_exp4_data(str(base_dir / "exp4"))

    print("\n全部数据生成完成!", file=sys.stderr)


if __name__ == "__main__":
    main()
