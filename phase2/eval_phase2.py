#!/usr/bin/env python
"""
eval_phase2 — Phase 2 实验评估框架

针对 4 个实验的统一评估入口，支持多种输入模式和评估指标。

实验:
  exp1 — 主动画图能力 (Active Diagram Generation)
  exp2 — 视觉记忆检索 (Visual Memory Retrieval)
  exp3 — 扫描效率     (Scan-vs-Read Efficiency)
  exp4 — 自发迁移     (Internalization Transfer)

用法:
  python eval_phase2.py exp1 --data ./test_data/exp1/
  python eval_phase2.py exp2 --data ./test_data/exp2/ --rounds 50,100,200
  python eval_phase2.py exp3 --data ./test_data/exp3/ --complexity simple,moderate,complex
  python eval_phase2.py exp4 --data ./test_data/exp4/
  python eval_phase2.py all   --data ./test_data/

输出:
  results/phase2_exp{N}_{timestamp}.json  — 完整结果
  results/phase2_exp{N}_{timestamp}.md   — 格式化报告
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, "results", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  评分器
# ═══════════════════════════════════════════════════════════════

def score_recall(response: str, ground_truth: list[str]) -> dict:
    """计算召回率分数。"""
    resp_lower = response.lower()
    found = [gt for gt in ground_truth if gt.lower() in resp_lower]
    return {
        "found": len(found),
        "total": len(ground_truth),
        "missing": [gt for gt in ground_truth if gt.lower() not in resp_lower],
        "precision": len(found) / len(ground_truth) if ground_truth else 1.0,
    }


def score_structural(response: str, expected_edges: list[dict]) -> dict:
    """评估结构理解准确性。"""
    correct = 0
    details = []
    for edge in expected_edges:
        from_node = edge.get("from", "").lower()
        to_node = edge.get("to", "").lower()
        rel_type = edge.get("type", "").lower()

        from_ok = from_node in response.lower()
        to_ok = to_node in response.lower()
        type_ok = rel_type in response.lower() if rel_type else True

        correct += 1 if (from_ok and to_ok and type_ok) else 0
        details.append({
            "edge": f"{from_node} -> {to_node}",
            "from_found": from_ok,
            "to_found": to_ok,
            "type_found": type_ok,
        })

    total = len(expected_edges)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0,
        "details": details,
    }


def detect_diagram_generation(response: str) -> dict:
    """检测回复中是否自发生成图示。"""
    has_mermaid = bool(re.search(r'```(mermaid|graph)', response, re.IGNORECASE))
    has_sketch = bool(re.search(r'<sketch>', response, re.IGNORECASE))
    has_structure = bool(re.search(
        r'(让我画|画一张|结构图|关系图|拓扑|箭头|节点|连线)',
        response
    ))
    has_structured_analysis = bool(re.search(
        r'(步骤\s*\d|第一步|第二步|首先.*其次|结构化分析|影响链路)',
        response
    ))

    # 尝试提取图描述
    diagram_content = ""
    if has_mermaid:
        m = re.search(r'```(?:mermaid|graph)\s*\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if m:
            diagram_content = m.group(1).strip()
    elif has_sketch:
        m = re.search(r'<sketch>(.*?)</sketch>', response, re.DOTALL)
        if m:
            diagram_content = m.group(1).strip()

    return {
        "has_diagram": has_mermaid or has_sketch,
        "has_structure_language": has_structure,
        "has_structured_analysis": has_structured_analysis,
        "diagram_content": diagram_content[:500] if diagram_content else "",
        "type": "mermaid" if has_mermaid else ("sketch" if has_sketch else "none"),
    }


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文 1 字≈1.5 token，英文 1 词≈1.3 token）。"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other = len(text) - chinese_chars - english_words
    return int(chinese_chars * 1.5 + english_words * 1.3 + other * 0.5)


# ═══════════════════════════════════════════════════════════════
#  LLM 调用封装 — 支持 云端API / 本地Ollama / 模拟
# ═══════════════════════════════════════════════════════════════

# 后端类型常量
BACKEND_CLOUD  = "cloud"   # OpenAI-compatible API
BACKEND_OLLAMA = "ollama"  # 本地 Ollama
BACKEND_MOCK   = "mock"    # 模拟（框架测试用）


def _detect_ollama() -> str | None:
    """检测本地 Ollama 是否可用，返回第一个可用模型名。"""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        if models:
            return models[0]["name"]
    except Exception:
        pass
    return None


def _list_ollama_models() -> list[str]:
    """列出本地 Ollama 所有模型。"""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


class ModelClient:
    """统一的模型调用接口。

    优先级: 云端API > 本地Ollama > 模拟
    可通过环境变量控制:
      EXPERIMENT_API_KEY / EXPERIMENT_API_BASE / EXPERIMENT_MODEL  → 云端
      EXPERIMENT_BACKEND=ollama  → 强制本地 Ollama
      EXPERIMENT_BACKEND=mock    → 模拟测试
    """

    def __init__(self, api_key: str = None, api_base: str = None,
                 model: str = None, backend: str = None):
        self.backend = backend or os.environ.get("EXPERIMENT_BACKEND", "")
        self.api_key = api_key or os.environ.get("EXPERIMENT_API_KEY", "")
        self.api_base = api_base or os.environ.get("EXPERIMENT_API_BASE", "")
        self.model = model or os.environ.get("EXPERIMENT_MODEL", "")
        self._ollama_model = None

        # 自动检测后端
        if not self.backend:
            if self.api_key and "your-key" not in self.api_key:
                self.backend = BACKEND_CLOUD
            elif _detect_ollama():
                self.backend = BACKEND_OLLAMA
                self._ollama_model = _detect_ollama()
            else:
                self.backend = BACKEND_MOCK

    def chat(self, messages: list[dict], temperature: float = 0.3,
             max_tokens: int = 2048, images: list[str] = None) -> tuple[str, float, int]:
        """返回 (响应文本, 耗时秒, 估算输入token数)。

        images: 图片文件路径列表（仅 cloud/ollama 后端支持）。
        """
        if self.backend == BACKEND_CLOUD:
            return self._chat_cloud(messages, temperature, max_tokens, images)
        elif self.backend == BACKEND_OLLAMA:
            return self._chat_ollama(messages, temperature, max_tokens, images)
        else:
            return self._chat_mock(messages)

    # ── 云端 API ──

    def _chat_cloud(self, messages, temp, max_tok, images) -> tuple[str, float, int]:
        import requests
        t0 = time.time()
        input_text = json.dumps(messages, ensure_ascii=False)
        input_tokens = estimate_tokens(input_text)

        # 如果有图片，构建多模态消息
        if images:
            content_parts = []
            for img_path in images:
                import base64
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            content_parts.append({"type": "text", "text": messages[-1]["content"]})
            messages = [{"role": "user", "content": content_parts}]

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": temp, "max_tokens": max_tok},
            timeout=120,
        )
        elapsed = time.time() - t0
        result = resp.json()
        return result["choices"][0]["message"]["content"], elapsed, input_tokens

    # ── 本地 Ollama ──

    def _chat_ollama(self, messages, temp, max_tok, images) -> tuple[str, float, int]:
        try:
            import ollama as ol
        except ImportError:
            print("  [Ollama] ollama-python 未安装，尝试 HTTP 调用", file=sys.stderr)
            return self._chat_ollama_http(messages, temp, max_tok, images)

        model = self._ollama_model or self.model or "qwen2.5:3b"
        t0 = time.time()
        input_text = json.dumps(messages, ensure_ascii=False)
        input_tokens = estimate_tokens(input_text)

        # 清理消息格式为 Ollama 兼容格式
        ollama_msgs = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                role = "user"
            ollama_msgs.append({"role": role, "content": str(m.get("content", ""))})

        kwargs = {"model": model, "messages": ollama_msgs,
                  "options": {"temperature": temp, "num_predict": max_tok}}
        if images:
            kwargs["images"] = images

        resp = ol.chat(**kwargs)
        elapsed = time.time() - t0
        return resp["message"]["content"], elapsed, input_tokens

    def _chat_ollama_http(self, messages, temp, max_tok, images) -> tuple[str, float, int]:
        """Fallback: 直接用 HTTP 调 Ollama API（不需要 ollama 包）。"""
        import requests
        model = self._ollama_model or self.model or "qwen2.5:3b"
        t0 = time.time()
        input_text = json.dumps(messages, ensure_ascii=False)
        input_tokens = estimate_tokens(input_text)

        body = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temp, "num_predict": max_tok},
            "stream": False,
        }
        if images:
            body["images"] = images

        resp = requests.post("http://localhost:11434/api/chat",
                            json=body, timeout=300)
        elapsed = time.time() - t0
        return resp.json()["message"]["content"], elapsed, input_tokens

    # ── 模拟（框架测试） ──

    def _chat_mock(self, messages: list[dict]) -> tuple[str, float, int]:
        input_text = json.dumps(messages, ensure_ascii=False)
        prompt = str(messages[-1].get("content", ""))[:200].lower()

        # 智能模拟：根据任务类型返回不同响应
        if "画" in prompt or "mermaid" in prompt or "结构" in prompt:
            resp = """<sketch>
```mermaid
graph TD
    A[修改模块] --> B[受影响的调用方1]
    A --> C[受影响的调用方2]
    B --> D[级联影响]
    style A fill:#ff4444
    style B fill:#ffd43b
    style C fill:#ffd43b
```
</sketch>

<analysis>
步骤1: 识别修改模块及其依赖关系
步骤2: 追踪影响链路：A → B/C → D
步骤3: 逐一修复受影响的调用方
</analysis>

修复方案: [模拟] 这是一个跨模块修改，需要关注所有调用方。"""
        elif "辩论" in prompt or "利益" in prompt or "决策" in prompt:
            resp = """首先，让我梳理一下利益相关方的关系：
1. 支持方 → 就业/税收
2. 反对方 → 环境/交通
3. 核心矛盾 → 短期经济 vs 长期发展

结构化分析: [模拟] 建议采用折中方案。"""
        elif "话题" in prompt or "总结" in prompt:
            resp = """话题演变图:
话题A → 话题B → 话题C → 回到话题A(第15轮)

关键决策:
- 第10轮: 确定使用Redis缓存
- [模拟] 其他决策..."""
        else:
            resp = """[模拟结构化分析]
步骤1: 先梳理当前情况
步骤2: 分析影响范围
步骤3: 给出建议

Q1: 6个节点 | Q2: 中心节点是B | Q3: 回路: 否 | Q4: 孤立节点: 是

[这是模拟响应，配置 Ollama 或 API key 获取真实结果]"""

        return (resp, 0.01, estimate_tokens(input_text))

    # ── 健康检查 ──

    def health_check(self) -> dict:
        """检查后端可用性。"""
        if self.backend == BACKEND_CLOUD:
            return {"backend": "cloud", "model": self.model,
                    "status": "enabled" if self.api_key else "no_api_key"}
        elif self.backend == BACKEND_OLLAMA:
            models = _list_ollama_models()
            return {"backend": "ollama", "model": self._ollama_model or self.model,
                    "status": "ready" if models else "no_models",
                    "available_models": models}
        else:
            return {"backend": "mock", "status": "mock_mode",
                    "note": "配置 Ollama 或 API key 以启用真实推理"}


# ═══════════════════════════════════════════════════════════════
#  实验 1: 主动画图能力
# ═══════════════════════════════════════════════════════════════

EXP1_PROMPT = """完成以下任务。如果你认为有必要先理清结构关系，请生成一个 Mermaid 图来描述当前的情况，然后再给出解决方案。

任务类型: {task_type}
任务描述:
{task_description}

上下文:
{context}"""


def run_exp1(data_dir: str, client: ModelClient) -> dict:
    """实验 1: 测试模型主动生成图的能力。"""
    print("=" * 60, file=sys.stderr)
    print("实验 1: 主动画图能力", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = []
    test_files = sorted(Path(data_dir).glob("*.json"))

    if not test_files:
        # 使用内置测试用例
        test_files = [_build_exp1_test("code"), _build_exp1_test("writing"), _build_exp1_test("conversation")]

    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            test = json.load(f)

        task_type = test.get("task_type", "unknown")
        task_desc = test.get("task_description", "")
        context = test.get("context", "")
        ground_truth = test.get("ground_truth", {})
        expected_files = ground_truth.get("expected_files", [])

        prompt = EXP1_PROMPT.format(
            task_type=task_type,
            task_description=task_desc,
            context=context,
        )

        messages = [{"role": "user", "content": prompt}]
        response, elapsed, input_tokens = client.chat(messages, temperature=0.3)

        diagram_info = detect_diagram_generation(response)
        recall = score_recall(response, expected_files) if expected_files else {}

        result = {
            "test_file": str(tf),
            "task_type": task_type,
            "diagram_generated": diagram_info["has_diagram"],
            "diagram_type": diagram_info["type"],
            "has_structure_language": diagram_info["has_structure_language"],
            "has_structured_analysis": diagram_info["has_structured_analysis"],
            "diagram_content": diagram_info["diagram_content"],
            "recall": recall,
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": input_tokens,
            "output_tokens": estimate_tokens(response),
            "response_preview": response[:300],
        }
        results.append(result)

        status = "✅ 生成了图" if diagram_info["has_diagram"] else "❌ 未生成图"
        print(f"  {task_type}: {status}", file=sys.stderr)

    # 汇总
    summary = _summarize_exp1(results)
    return {"experiment": "exp1_active_drawing", "results": results, "summary": summary}


def _summarize_exp1(results: list[dict]) -> dict:
    gen_count = sum(1 for r in results if r["diagram_generated"])
    struct_count = sum(1 for r in results if r["has_structured_analysis"])
    total = len(results)

    recalls = [r["recall"].get("precision", 0) for r in results if r["recall"]]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    gen_recalls = [r["recall"].get("precision", 0) for r in results
                   if r["diagram_generated"] and r["recall"]]
    no_gen_recalls = [r["recall"].get("precision", 0) for r in results
                      if not r["diagram_generated"] and r["recall"]]
    avg_gen_recall = sum(gen_recalls) / len(gen_recalls) if gen_recalls else 0
    avg_no_gen_recall = sum(no_gen_recalls) / len(no_gen_recalls) if no_gen_recalls else 0

    return {
        "total_cases": total,
        "generation_rate": f"{gen_count}/{total} ({gen_count/total*100:.0f}%)" if total else "0",
        "structured_analysis_rate": f"{struct_count}/{total} ({struct_count/total*100:.0f}%)" if total else "0",
        "avg_recall": round(avg_recall, 3),
        "avg_recall_with_diagram": round(avg_gen_recall, 3),
        "avg_recall_without_diagram": round(avg_no_gen_recall, 3),
    }


def _build_exp1_test(task_type: str) -> Path:
    """构建内置测试用例。"""
    tests = {
        "code": {
            "task_type": "code_modification",
            "task_description": "修改 Task.to_dict() 的返回值格式，从返回 dict 改为返回 JSON 字符串。确保所有调用方同步更新。",
            "context": "项目包含 models/task.py, services/task_service.py, api/handlers.py, utils/validators.py。task_service.py 和 handlers.py 调用了 to_dict()。",
            "ground_truth": {"expected_files": [
                "models/task.py", "services/task_service.py", "api/handlers.py", "utils/validators.py"
            ]},
        },
        "writing": {
            "task_type": "article_outline",
            "task_description": "分析以下文章草稿的结构，指出论点之间的逻辑关系和不一致之处。",
            "context": "文章讨论 AI 在医疗中的应用，提出三个论点：(1) AI 诊断准确率超过医生；(2) 但 AI 缺乏同理心影响患者信任；(3) 因此需要人机协作模式，但实施面临数据隐私障碍。",
            "ground_truth": {
                "expected_files": [
                    "论点1: AI诊断", "论点2: 同理心问题", "论点3: 人机协作",
                    "矛盾: 论点1 vs 论点2", "障碍: 隐私"
                ],
            },
        },
        "conversation": {
            "task_type": "conversation_summary",
            "task_description": "总结以下对话中讨论的所有话题及其演变过程。",
            "context": "用户和 AI 讨论了三件事：代码重构方案（前10轮）→ 测试策略（涉及单元测试和集成测试取舍，11-20轮）→ 部署流程（CI/CD 配置，21-30轮）。用户在第15轮曾回到重构话题提出新想法。",
            "ground_truth": {
                "expected_files": [
                    "话题1: 重构", "话题2: 测试策略", "话题3: 部署流程",
                    "交叉: 重构→测试 (round 15)", "演进: 测试→部署"
                ],
            },
        },
    }

    test = tests.get(task_type, tests["code"])
    out_path = Path(RESULTS_DIR) / f"exp1_test_{task_type}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    return out_path


# ═══════════════════════════════════════════════════════════════
#  实验 2: 视觉记忆检索
# ═══════════════════════════════════════════════════════════════

def run_exp2(data_dir: str, client: ModelClient, round_levels: list[int] = None) -> dict:
    """实验 2: 测试图辅助记忆 vs 纯文本重新阅读的检索效率。"""
    print("=" * 60, file=sys.stderr)
    print("实验 2: 视觉记忆检索", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    round_levels = round_levels or [50, 100, 200]
    results = []

    for n_rounds in round_levels:
        conv_path = Path(data_dir) / f"conversation_{n_rounds}.json"
        graph_path = Path(data_dir) / f"topic_graph_{n_rounds}.png"

        if not conv_path.exists():
            print(f"  跳过: 对话文件不存在 {conv_path}", file=sys.stderr)
            continue

        with open(conv_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)

        conversation = conv_data.get("conversation", [])
        questions = conv_data.get("questions", [])
        fact_questions = [q for q in questions if q.get("type") == "fact"]
        relation_questions = [q for q in questions if q.get("type") == "relation"]

        # ── 条件 1: 纯文本 ──
        print(f"  {n_rounds}轮 — 条件1: 纯文本", file=sys.stderr)
        text_results = _run_retrieval_condition(
            client, conversation, questions, mode="text_only", graph_path=None
        )

        # ── 条件 2: 纯图 ──
        print(f"  {n_rounds}轮 — 条件2: 纯图", file=sys.stderr)
        graph_results = _run_retrieval_condition(
            client, conversation, questions, mode="graph_only", graph_path=str(graph_path)
            if graph_path.exists() else None
        )

        # ── 条件 3: 先图后文 ──
        print(f"  {n_rounds}轮 — 条件3: 先图后文", file=sys.stderr)
        hybrid_results = _run_retrieval_condition(
            client, conversation, questions, mode="graph_then_text", graph_path=str(graph_path)
            if graph_path.exists() else None
        )

        results.append({
            "rounds": n_rounds,
            "fact_questions": len(fact_questions),
            "relation_questions": len(relation_questions),
            "text_only": text_results,
            "graph_only": graph_results,
            "graph_then_text": hybrid_results,
        })

    summary = _summarize_exp2(results)
    return {"experiment": "exp2_visual_memory", "results": results, "summary": summary}


def _run_retrieval_condition(client: ModelClient, conversation: list,
                             questions: list, mode: str,
                             graph_path: str = None) -> dict:
    """运行一种检索条件。"""
    if mode == "text_only":
        conv_text = "\n".join(
            f"[Round {m.get('round','?')}] {m.get('role','?')}: {m.get('content','')[:200]}"
            for m in conversation
        )
        prompt = f"阅读以下对话，回答 {len(questions)} 个问题。\n\n{conv_text}\n\n"
    elif mode == "graph_only":
        prompt = "请根据之前提供的话题关系图（PNG），回答以下问题。注意：你只能依据图中的信息。\n\n"
    elif mode == "graph_then_text":
        conv_text = "\n".join(
            f"[Round {m.get('round','?')}] {m.get('role','?')}: {m.get('content','')[:200]}"
            for m in conversation
        )
        prompt = f"请先浏览话题关系图了解整体结构，然后选择性阅读以下对话回答 {len(questions)} 个问题。\n\n{conv_text}\n\n"

    for i, q in enumerate(questions):
        prompt += f"Q{i+1}: {q.get('question', '')}\n"

    prompt += "\n请用简洁格式回答: Q1: 答案 | Q2: 答案 | ..."

    # 对于含图的模式，图需要上层加载
    # 这里仅构建文本提示；实际含图调用由上层处理
    messages = [{"role": "user", "content": prompt}]
    response, elapsed, input_tokens = client.chat(messages, temperature=0.2)

    # 解析答案
    answers = _parse_qa_response(response, len(questions))
    qa_results = []
    for i, q in enumerate(questions):
        expected = q.get("answer", [])
        pred = answers.get(i, "")
        score = score_recall(pred, expected) if isinstance(expected, list) else \
               score_recall(pred, [expected])
        qa_results.append({"question": q.get("question", ""), "score": score})

    avg_recall = sum(r["score"]["precision"] for r in qa_results) / len(qa_results) if qa_results else 0

    return {
        "mode": mode,
        "avg_recall": round(avg_recall, 3),
        "input_tokens": input_tokens,
        "output_tokens": estimate_tokens(response),
        "elapsed_seconds": round(elapsed, 2),
        "efficiency": round(avg_recall / max(input_tokens, 1) * 1000, 4),  # recall per 1K tokens
        "qa_results": qa_results,
    }


def _parse_qa_response(response: str, num_questions: int) -> dict[int, str]:
    """解析 Q&A 响应。"""
    answers = {}
    for i in range(num_questions):
        pattern = rf'Q{i+1}[：:]\s*(.+?)(?=Q{i+2}[：:]|\Z)'
        m = re.search(pattern, response, re.DOTALL)
        if m:
            answers[i] = m.group(1).strip()
    return answers


def _summarize_exp2(results: list[dict]) -> dict:
    summary_rows = []
    for r in results:
        for mode in ["text_only", "graph_only", "graph_then_text"]:
            mode_results = r.get(mode, {})
            summary_rows.append({
                "rounds": r["rounds"],
                "mode": mode,
                "avg_recall": mode_results.get("avg_recall", 0),
                "tokens": mode_results.get("input_tokens", 0),
                "efficiency": mode_results.get("efficiency", 0),
            })

    # 按轮数分组计算效率比
    efficiency_by_rounds = defaultdict(dict)
    for row in summary_rows:
        rounds = row["rounds"]
        mode = row["mode"]
        efficiency_by_rounds[rounds][mode] = row["efficiency"]

    comparisons = {}
    for rounds, modes in efficiency_by_rounds.items():
        text_eff = modes.get("text_only", 0)
        hybrid_eff = modes.get("graph_then_text", 0)
        if text_eff > 0:
            comparisons[f"efficiency_ratio_{rounds}r"] = round(hybrid_eff / text_eff, 2)

    return {
        "summary_rows": summary_rows,
        "efficiency_comparison": comparisons,
        "total_conditions": len(summary_rows),
    }


# ═══════════════════════════════════════════════════════════════
#  实验 3: 扫描效率
# ═══════════════════════════════════════════════════════════════

def run_exp3(data_dir: str, client: ModelClient,
             complexity_levels: list[str] = None) -> dict:
    """实验 3: 测试不同处理深度下的扫描效率。"""
    print("=" * 60, file=sys.stderr)
    print("实验 3: 扫描效率", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    complexity_levels = complexity_levels or ["simple", "moderate", "complex"]
    results = []

    for level in complexity_levels:
        data_path = Path(data_dir) / f"graph_{level}.json"
        if not data_path.exists():
            # 使用内置用例
            data_path = _build_exp3_test(level)

        with open(data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        tasks = test_data.get("tasks", [])
        graph_path = Path(data_dir) / f"graph_{level}.png"

        # ── 条件 1: 扫视（限制时间思考） ──
        scan_prompt = """你只能快速扫视这张图（3秒），然后凭印象回答以下问题。不要仔细检查。

图中有多少个节点？
哪个节点连接最多？
有没有回路/循环？
有没有孤立节点？

请用简洁格式回答: 节点数: X | 中心节点: Y | 回路: 是/否 | 孤立节点: 是/否"""
        scan_resp, scan_time, scan_tokens = client.chat(
            [{"role": "user", "content": scan_prompt}], temperature=0.1, max_tokens=256
        )

        # ── 条件 2: 仔细看图 ──
        full_prompt = """请仔细查看这张图，准确回答以下问题:

1. 一共有多少个节点？
2. 哪个节点连接最多（度数最高）？
3. 图中有没有回路/循环？
4. 有没有孤立节点（没有任何连接的节点）？
5. 从节点A到节点B的最短路径是什么？
"""
        full_resp, full_time, full_tokens = client.chat(
            [{"role": "user", "content": full_prompt}], temperature=0.1, max_tokens=512
        )

        # ── 条件 3: 纯文本描述 ──
        text_desc = test_data.get("text_description", "")
        text_prompt = f"""阅读以下图结构的文本描述，然后回答问题:

{text_desc}

问题:
1. 一共有多少个节点？
2. 哪个节点连接最多？
3. 图中有没有回路/循环？
4. 有没有孤立节点？
"""
        text_resp, text_time, text_tokens = client.chat(
            [{"role": "user", "content": text_prompt}], temperature=0.1, max_tokens=512
        )

        # 评分
        ground_truth = test_data.get("ground_truth", {})
        scan_scores = _score_graph_questions(scan_resp, ground_truth)
        full_scores = _score_graph_questions(full_resp, ground_truth)
        text_scores = _score_graph_questions(text_resp, ground_truth)

        level_result = {
            "complexity": level,
            "node_count": ground_truth.get("node_count", 0),
            "edge_count": ground_truth.get("edge_count", 0),
            "scan": {
                "accuracy": scan_scores,
                "tokens": scan_tokens,
                "time": round(scan_time, 2),
                "efficiency": round(scan_scores.get("avg_accuracy", 0) / max(scan_tokens, 1) * 1000, 4),
            },
            "full_read": {
                "accuracy": full_scores,
                "tokens": full_tokens,
                "time": round(full_time, 2),
                "efficiency": round(full_scores.get("avg_accuracy", 0) / max(full_tokens, 1) * 1000, 4),
            },
            "text_only": {
                "accuracy": text_scores,
                "tokens": text_tokens,
                "time": round(text_time, 2),
                "efficiency": round(text_scores.get("avg_accuracy", 0) / max(text_tokens, 1) * 1000, 4),
            },
        }
        results.append(level_result)
        print(f"  {level}: scan={scan_scores.get('avg_accuracy',0):.2f} "
              f"full={full_scores.get('avg_accuracy',0):.2f} "
              f"text={text_scores.get('avg_accuracy',0):.2f}", file=sys.stderr)

    summary = _summarize_exp3(results)
    return {"experiment": "exp3_scan_efficiency", "results": results, "summary": summary}


def _score_graph_questions(response: str, ground_truth: dict) -> dict:
    """对图理解问题进行评分。"""
    scores = {}

    # 节点数
    gt_nodes = ground_truth.get("node_count", 0)
    node_match = re.search(r'(\d+)\s*(个)?(节点|node)', response, re.IGNORECASE)
    if node_match:
        pred_nodes = int(node_match.group(1))
        scores["node_count"] = 1.0 if pred_nodes == gt_nodes else (
            0.5 if abs(pred_nodes - gt_nodes) <= max(1, gt_nodes * 0.2) else 0.0)
    else:
        scores["node_count"] = 0.0

    # 中心节点
    gt_center = ground_truth.get("center_node", "").lower()
    if gt_center and gt_center in response.lower():
        scores["center_node"] = 1.0
    else:
        scores["center_node"] = 0.5 if gt_center else 0.0  # 部分正确

    # 回路
    gt_cycle = ground_truth.get("has_cycle", False)
    has_cycle_pred = bool(re.search(r'(有|是|yes|存在).*(循环|回路|cycle)', response, re.IGNORECASE))
    no_cycle_pred = bool(re.search(r'(没有|无|否|no).*(循环|回路|cycle)', response, re.IGNORECASE))
    if gt_cycle and has_cycle_pred:
        scores["cycle"] = 1.0
    elif not gt_cycle and no_cycle_pred:
        scores["cycle"] = 1.0
    elif not gt_cycle and not no_cycle_pred and not has_cycle_pred:
        scores["cycle"] = 0.5
    else:
        scores["cycle"] = 0.0

    # 孤立节点
    gt_isolated = ground_truth.get("has_isolated", False)
    has_iso_pred = bool(re.search(r'(有|是|yes|存在).*(孤立|isolated)', response, re.IGNORECASE))
    no_iso_pred = bool(re.search(r'(没有|无|否|no).*(孤立|isolated)', response, re.IGNORECASE))
    if gt_isolated and has_iso_pred:
        scores["isolated"] = 1.0
    elif not gt_isolated and no_iso_pred:
        scores["isolated"] = 1.0
    elif not gt_isolated and not no_iso_pred and not has_iso_pred:
        scores["isolated"] = 0.5
    else:
        scores["isolated"] = 0.0

    values = list(scores.values())
    scores["avg_accuracy"] = round(sum(values) / len(values), 3) if values else 0
    return scores


def _build_exp3_test(level: str) -> Path:
    """构建内置扫描效率测试用例。"""
    levels = {
        "simple": {
            "node_count": 6, "edge_count": 7, "has_cycle": False, "has_isolated": True,
            "center_node": "B",
            "text_description": "图有6个节点: A连接到B和C, B连接到A/C/D, C连接到B, D连接到B, E孤立无连接, F连接到D。",
        },
        "moderate": {
            "node_count": 15, "edge_count": 22, "has_cycle": True, "has_isolated": False,
            "center_node": "核心模块",
            "text_description": "图有15个节点，核心模块连接了7个其他模块，存在 A→B→C→A 的反馈回路。",
        },
        "complex": {
            "node_count": 30, "edge_count": 55, "has_cycle": True, "has_isolated": True,
            "center_node": "主服务",
            "text_description": "图有30个节点和55条边，主服务节点度数最高（12），存在多处反馈回路，有2个孤立节点。",
        },
    }

    data = levels.get(level, levels["simple"])
    tasks = [{"type": "count"}, {"type": "center"}, {"type": "cycle"}, {"type": "isolated"}]
    test = {"complexity": level, "ground_truth": data, "tasks": tasks,
            "text_description": data.pop("text_description", "")}

    out_path = Path(RESULTS_DIR) / f"exp3_test_{level}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)
    return out_path


# ═══════════════════════════════════════════════════════════════
#  实验 4: 自发迁移
# ═══════════════════════════════════════════════════════════════

EXP4_TASKS = {
    "decision_analysis": {
        "prompt": """分析以下多方利益相关者的决策场景。请给出你的分析和建议。

场景: 一个城市需要决定是否在市中心建设新的商业综合体。支持方认为可以创造就业和税收；居民担心交通拥堵和噪音；环保组织指出绿地减少；小商户害怕被大型连锁取代。市政府需要在三个月内做决定。""",
        "evaluation": {
            "expected_stakeholders": ["政府", "居民", "环保组织", "小商户", "开发商"],
            "expected_relationships": ["支持vs反对", "经济利益vs环境", "短期vs长期"],
        },
    },
    "argument_mapping": {
        "prompt": """分析以下辩论的观点关系和论证链条。

辩论: 关于"是否应该全面禁止校园手机使用"
- A方: 手机分散注意力，导致成绩下降（引用PISA数据）
- B方: 手机是数字素养工具，完全禁止会让学生落后于时代
- A方反驳: 数字素养可以在计算机课上培养，不需要手机
- B方提出: 紧急情况下学生需要联系家长
- A方回应: 学校有足够的固定电话和教师通讯设备""",
        "evaluation": {
            "expected_arguments": ["注意力分散", "数字素养", "紧急联系", "替代方案"],
            "expected_structure": "有链条证据 A→反驳→再反驳",
        },
    },
    "knowledge_graph": {
        "prompt": """根据以下事实，回答"哪些实体与张教授有间接关系？"

事实:
- 张教授是AI实验室主任
- AI实验室位于计算机学院
- 计算机学院与数学学院合作
- 数学学院的李教授研究图论
- 图论是张教授课题的基础
- 该课题由王博士负责
- 王博士曾是张教授的学生""",
        "evaluation": {
            "expected_entities": ["李教授", "图论", "王博士", "数学学院"],
            "expected_paths": ["张教授→AI实验室→计算机学院→数学学院→李教授"],
        },
    },
}


def run_exp4(data_dir: str, client: ModelClient) -> dict:
    """实验 4: 测试自发结构化思维迁移。"""
    print("=" * 60, file=sys.stderr)
    print("实验 4: 自发迁移", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = []

    # 优先使用外部数据，否则使用内置
    ext_path = Path(data_dir)
    if ext_path.exists() and list(ext_path.glob("*.json")):
        with open(ext_path / "exp4_tasks.json" if (ext_path / "exp4_tasks.json").exists()
                  else next(ext_path.glob("*.json")), "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        tasks = EXP4_TASKS

    for task_name, task_data in tasks.items():
        prompt = task_data.get("prompt", "")
        evaluation = task_data.get("evaluation", {})

        messages = [{"role": "user", "content": prompt}]
        response, elapsed, input_tokens = client.chat(messages, temperature=0.3)

        diagram_info = detect_diagram_generation(response)
        struct_info = _detect_structured_reasoning(response)

        # 评分
        expected_items = evaluation.get("expected_stakeholders", evaluation.get("expected_arguments",
                               evaluation.get("expected_entities", [])))
        recall = score_recall(response, expected_items)

        result = {
            "task": task_name,
            "diagram_generated": diagram_info["has_diagram"],
            "has_structured_analysis": struct_info["has_structure"],
            "reasoning_patterns": struct_info["patterns"],
            "recall": recall,
            "input_tokens": input_tokens,
            "output_tokens": estimate_tokens(response),
            "elapsed_seconds": round(elapsed, 2),
            "response_preview": response[:300],
        }
        results.append(result)

        status = "✅ 结构化" if struct_info["has_structure"] else "❌ 未结构化"
        print(f"  {task_name}: {status} | recall={recall.get('precision',0):.2f}", file=sys.stderr)

    summary = _summarize_exp4(results)
    return {"experiment": "exp4_internalization", "results": results, "summary": summary}


def _detect_structured_reasoning(response: str) -> dict:
    """检测结构化推理模式。"""
    patterns = []
    if re.search(r'(步骤|第一步|第二步|首先.*其次|第一.*第二)', response):
        patterns.append("numbered_steps")
    if re.search(r'(关系|关联|连接|依赖|影响).*(图|结构|链路)', response):
        patterns.append("relationship_language")
    if re.search(r'(↑|↓|→|←|⇒|⟹)', response):
        patterns.append("arrow_notation")
    if re.search(r'```(mermaid|graph)', response, re.IGNORECASE):
        patterns.append("mermaid_diagram")
    if re.search(r'(树状|层级|层次|拓扑|节点|边)', response):
        patterns.append("graph_terms")
    if re.search(r'(先.*分析|先.*理清|先.*梳理)', response):
        patterns.append("analyze_first")

    return {
        "has_structure": len(patterns) > 0,
        "pattern_count": len(patterns),
        "patterns": patterns,
    }


def _summarize_exp4(results: list[dict]) -> dict:
    struct_count = sum(1 for r in results if r["has_structured_analysis"])
    gen_count = sum(1 for r in results if r["diagram_generated"])
    total = len(results)

    pattern_dist = defaultdict(int)
    for r in results:
        for p in r.get("reasoning_patterns", []):
            pattern_dist[p] += 1

    recalls = [r["recall"].get("precision", 0) for r in results]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    struct_recalls = [r["recall"].get("precision", 0) for r in results
                      if r["has_structured_analysis"]]
    no_struct_recalls = [r["recall"].get("precision", 0) for r in results
                         if not r["has_structured_analysis"]]
    avg_struct_recall = sum(struct_recalls) / len(struct_recalls) if struct_recalls else 0
    avg_no_struct_recall = sum(no_struct_recalls) / len(no_struct_recalls) if no_struct_recalls else 0

    return {
        "total_tasks": total,
        "structured_rate": f"{struct_count}/{total} ({struct_count/total*100:.0f}%)" if total else "0",
        "diagram_generation_rate": f"{gen_count}/{total} ({gen_count/total*100:.0f}%)" if total else "0",
        "pattern_distribution": dict(pattern_dist),
        "avg_recall": round(avg_recall, 3),
        "avg_recall_with_structure": round(avg_struct_recall, 3),
        "avg_recall_without_structure": round(avg_no_struct_recall, 3),
    }


# ═══════════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════════

def generate_report(all_results: dict) -> str:
    """生成 Markdown 评估报告。"""
    lines = ["# Phase 2 实验评估报告\n"]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    for exp_name, exp_data in all_results.items():
        lines.append(f"## {exp_name}\n")
        summary = exp_data.get("summary", {})

        if exp_name == "exp1_active_drawing":
            lines.append(f"- 总用例: {summary.get('total_cases', 0)}")
            lines.append(f"- 图生成率: {summary.get('generation_rate', 'N/A')}")
            lines.append(f"- 结构化分析率: {summary.get('structured_analysis_rate', 'N/A')}")
            lines.append(f"- 有图时平均召回: {summary.get('avg_recall_with_diagram', 0):.3f}")
            lines.append(f"- 无图时平均召回: {summary.get('avg_recall_without_diagram', 0):.3f}")
            lines.append("")

        elif exp_name == "exp2_visual_memory":
            lines.append("| 轮数 | 模式 | 召回率 | Token数 | 效率 (recall/1K) |")
            lines.append("|------|------|--------|---------|------------------|")
            for row in summary.get("summary_rows", []):
                lines.append(
                    f"| {row['rounds']} | {row['mode']} | {row['avg_recall']:.3f} | "
                    f"{row['tokens']} | {row['efficiency']:.4f} |"
                )
            lines.append("")
            for key, val in summary.get("efficiency_comparison", {}).items():
                lines.append(f"- {key}: {val:.2f}x")
            lines.append("")

        elif exp_name == "exp3_scan_efficiency":
            lines.append("| 复杂度 | 模式 | 平均准确率 | Token数 | 效率 |")
            lines.append("|--------|------|-----------|---------|------|")
            for r in exp_data.get("results", []):
                for mode in ["scan", "full_read", "text_only"]:
                    mdata = r.get(mode, {})
                    acc = mdata.get("accuracy", {})
                    lines.append(
                        f"| {r['complexity']} | {mode} | {acc.get('avg_accuracy', 0):.3f} | "
                        f"{mdata.get('tokens', 0)} | {mdata.get('efficiency', 0):.4f} |"
                    )
            lines.append("")

        elif exp_name == "exp4_internalization":
            lines.append(f"- 总任务: {summary.get('total_tasks', 0)}")
            lines.append(f"- 结构化率: {summary.get('structured_rate', 'N/A')}")
            lines.append(f"- 图生成率: {summary.get('diagram_generation_rate', 'N/A')}")
            lines.append(f"- 有结构时召回: {summary.get('avg_recall_with_structure', 0):.3f}")
            lines.append(f"- 无结构时召回: {summary.get('avg_recall_without_structure', 0):.3f}")
            lines.append(f"- 推理模式分布: {summary.get('pattern_distribution', {})}")
            lines.append("")

    lines.append("---\n")
    lines.append("*由 eval_phase2.py 自动生成*")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 2 实验评估框架")
    parser.add_argument("experiment", choices=["exp1", "exp2", "exp3", "exp4", "all"],
                        help="要运行的实验编号")
    parser.add_argument("--data", "-d", default="./test_data/",
                        help="测试数据目录")
    parser.add_argument("--rounds", "-r", default="50,100,200",
                        help="Exp2: 对话轮数 (逗号分隔)")
    parser.add_argument("--complexity", "-c", default="simple,moderate,complex",
                        help="Exp3: 复杂度级别 (逗号分隔)")
    parser.add_argument("--output", "-o", default=None,
                        help="结果输出路径 (默认: results/phase2/)")
    args = parser.parse_args()

    client = ModelClient()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) if args.output else Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    experiments = ["exp1", "exp2", "exp3", "exp4"] if args.experiment == "all" else [args.experiment]

    for exp in experiments:
        try:
            if exp == "exp1":
                result = run_exp1(args.data, client)
            elif exp == "exp2":
                rounds = [int(r.strip()) for r in args.rounds.split(",")]
                result = run_exp2(args.data, client, rounds)
            elif exp == "exp3":
                levels = [l.strip() for l in args.complexity.split(",")]
                result = run_exp3(args.data, client, levels)
            elif exp == "exp4":
                result = run_exp4(args.data, client)
            else:
                continue

            all_results[exp] = result

            # 保存 JSON 结果
            json_path = output_dir / f"phase2_{exp}_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n结果保存: {json_path}", file=sys.stderr)

        except Exception as e:
            print(f"\n[错误] {exp} 执行失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # 生成综合报告
    if all_results:
        report = generate_report(all_results)
        report_path = output_dir / f"phase2_report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告: {report_path}", file=sys.stderr)
        print(report)


if __name__ == "__main__":
    main()
