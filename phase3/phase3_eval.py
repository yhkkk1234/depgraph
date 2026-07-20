#!/usr/bin/env python
"""
phase3_eval — Phase 3 图形记忆系统评估

支持三种评估模式:
  1. api 模式: 调用云端多模态 API (GPT-4o/Gemini/Claude/...)
     需要设置环境变量: EXPERIMENT_API_KEY, EXPERIMENT_API_BASE, EXPERIMENT_MODEL
  2. ollama 模式: 调用本地 Ollama 多模态模型
  3. offline 模式: 关键词模拟 (快速测试管道, 不依赖 API)

评估实验:
  Exp1: 扫图 vs 读全文 (3条件对比)
  Exp2: 增量记忆保持率
  Exp3: 标注效果消融 (4版本对比)
  Exp4: 规模扩展优势曲线
  Exp5: 训练后主动生成率
  Exp6: 零样本转移率

用法:
  # 本机用 API 评估 (推荐用于 Phase 3a)
  set EXPERIMENT_API_KEY=your-key
  set EXPERIMENT_API_BASE=https://api.openai.com/v1
  set EXPERIMENT_MODEL=gpt-4o
  python phase3_eval.py --all --data ./phase3_data/ --mode api

  # AutoDL 本地 Ollama 评估
  python phase3_eval.py --all --data ./phase3_data/ --mode ollama

  # 离线快速测试
  python phase3_eval.py --all --data ./phase3_data/ --mode offline
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_tokens(text: str) -> int:
    """粗略 token 估算: 中文 1 char≈1.5 token, 英文 1 word≈1.3 token"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return int(chinese_chars * 1.5 + english_words * 1.3 + len(text) * 0.3)


def simple_match_score(predicted: str, ground_truth: str) -> float:
    """简单语义匹配评分 (关键词重叠)"""
    if not predicted or not ground_truth:
        return 0.0
    pred_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', predicted.lower()))
    gt_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{2,}', ground_truth.lower()))
    if not gt_words:
        return 1.0 if not pred_words else 0.0
    overlap = len(pred_words & gt_words)
    return overlap / len(gt_words)


def fuzzy_match(predicted: str, ground_truth: str) -> dict:
    """模糊匹配评分，返回详情"""
    score = simple_match_score(predicted, ground_truth)
    return {
        "score": round(score, 4),
        "pred_tokens": count_tokens(predicted),
        "gt_tokens": count_tokens(ground_truth),
        "pred_length": len(predicted),
        "gt_length": len(ground_truth),
    }


# ═══════════════════════════════════════════════════════════════
#  Exp1 评估: 静态记忆图 — 三维度对比
# ═══════════════════════════════════════════════════════════════

def eval_exp1(data_path: str, mode: str = "offline"):
    """评估 Exp1: 扫图 vs 读全文 vs 混合"""
    data = load_json(data_path)
    results = {
        "experiment": "Exp1_StaticMemory",
        "mode": mode,
        "total_samples": len(data),
        "conditions": {
            "scan_graph": {"scores": [], "tokens": [], "correct": 0, "total": 0},
            "read_full_text": {"scores": [], "tokens": [], "correct": 0, "total": 0},
            "hybrid": {"scores": [], "tokens": [], "correct": 0, "total": 0},
        },
        "by_question_type": defaultdict(lambda: defaultdict(list)),
        "per_sample": [],
    }

    for sample in data:
        memory_graph = sample.get("memory_graph", {})
        conversation = sample.get("conversation", [])
        conv_text = _conversation_to_text(conversation)
        graph_text = _graph_to_text(memory_graph)
        graph_tokens = count_tokens(graph_text)
        conv_tokens = count_tokens(conv_text)

        sample_result = {
            "sample_id": sample.get("sample_id", "?"),
            "domain": sample.get("domain", "?"),
            "num_rounds": sample.get("num_rounds", 0),
            "qa_results": [],
        }

        for qa in sample.get("qa_pairs", []):
            qtype = qa.get("type", "fact")
            gt = qa.get("answer", "")
            question = qa.get("question", "")

            scan_pred = ask_model(question, graph_text, "scan", _find_graph_png(data_path, sample))
            text_pred = ask_model(question, conv_text, "read")
            hybrid_pred = ask_model(question, graph_text + "\n" + conv_text[:conv_tokens//3], "hybrid", _find_graph_png(data_path, sample))

            scan_result = fuzzy_match(scan_pred, gt)
            text_result = fuzzy_match(text_pred, gt)
            hybrid_result = fuzzy_match(hybrid_pred, gt)

            for cond, result in [("scan_graph", scan_result), ("read_full_text", text_result), ("hybrid", hybrid_result)]:
                results["conditions"][cond]["scores"].append(result["score"])
                results["conditions"][cond]["total"] += 1
                if result["score"] > 0.3:
                    results["conditions"][cond]["correct"] += 1

                results["by_question_type"][qtype][cond].append(result["score"])

            sample_result["qa_results"].append({
                "question": question[:60],
                "type": qtype,
                "scan_score": scan_result["score"],
                "text_score": text_result["score"],
                "hybrid_score": hybrid_result["score"],
            })

        results["per_sample"].append(sample_result)

    for cond in results["conditions"]:
        scores = results["conditions"][cond]["scores"]
        total = results["conditions"][cond]["total"]
        results["conditions"][cond]["mean_score"] = round(sum(scores) / len(scores), 4) if scores else 0
        results["conditions"][cond]["accuracy"] = round(results["conditions"][cond]["correct"] / total, 4) if total > 0 else 0

    for qtype in results["by_question_type"]:
        for cond in list(results["by_question_type"][qtype].keys()):
            scores = results["by_question_type"][qtype][cond]
            results["by_question_type"][qtype][f"{cond}_mean"] = round(sum(scores) / len(scores), 4) if scores else 0

    results["token_comparison"] = {
        "graph_avg_tokens": sum(count_tokens(_graph_to_text(s.get("memory_graph", {}))) for s in data) // max(1, len(data)),
        "text_avg_tokens": sum(count_tokens(_conversation_to_text(s.get("conversation", []))) for s in data) // max(1, len(data)),
    }

    return results


def _conversation_to_text(conversation: list[dict]) -> str:
    lines = []
    for msg in conversation:
        role = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"[轮{msg.get('round', '?')}] {role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _graph_to_text(memory_graph: dict) -> str:
    """将记忆图转为纯文本描述 (模拟文字 Key 内容)"""
    nodes = memory_graph.get("nodes", [])
    edges = memory_graph.get("edges", [])
    lines = ["# 记忆图文字描述"]
    for n in nodes:
        label = n.get("label", n["id"])
        imp = n.get("importance", "normal")
        desc = n.get("description", "")
        annotations = n.get("annotations", [])
        lines.append(f"\n[{imp}] {label}")
        if desc:
            lines.append(f"  摘要: {desc}")
        for ann in annotations:
            lines.append(f"  标注: {ann.get('type')}: {ann.get('text', '')}")
    for e in edges[:10]:
        lines.append(f"\n{e['from']} --({e.get('type', 'ref')})--> {e['to']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  EvaluatorAPI — 多模态模型调用 (api / ollama / mock)
# ═══════════════════════════════════════════════════════════════

class EvaluatorAPI:
    """多模态 API 评估器，支持三种后端。

    后端检测优先级:
      1. EXPERIMENT_API_KEY 已配置 → api
      2. Ollama 可用 → ollama
      3. 否则 → mock (快速管道测试)
    """

    def __init__(self, backend: str = None):
        self.backend = backend or self._detect_backend()
        self.api_key = os.environ.get("EXPERIMENT_API_KEY", "")
        self.api_base = os.environ.get("EXPERIMENT_API_BASE", "https://api.openai.com/v1")
        self.model = os.environ.get("EXPERIMENT_MODEL", "qwen3.7-plus")
        self._calls = 0
        self._total_time = 0.0
        self._total_tokens = 0

    @staticmethod
    def _detect_backend() -> str:
        if os.environ.get("EXPERIMENT_API_KEY", "").strip():
            return "api"
        try:
            import ollama
            models = ollama.list()
            if models and len(models.get("models", [])) > 0:
                return "ollama"
        except Exception:
            pass
        return "mock"

    def ask_with_graph(self, question: str, graph_text: str,
                       graph_png_path: str = None, mode: str = "scan",
                       max_retries: int = 2) -> str:
        """向模型提问并返回回答。含重试机制。"""

        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                if self.backend == "api":
                    result = self._call_api(question, graph_text, graph_png_path, mode)
                elif self.backend == "ollama":
                    result = self._call_ollama(question, graph_text, graph_png_path, mode)
                else:
                    result = self._call_mock(question, graph_text, mode)

                self._calls += 1
                self._total_time += time.time() - t0
                return result

            except Exception as e:
                msg = str(e)[:80]
                if attempt < max_retries and ("timeout" in msg.lower() or "connection" in msg.lower()):
                    wait = (attempt + 1) * 10
                    print(f"  [RETRY] attempt {attempt+1} failed ({msg}), waiting {wait}s...",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                # 最后一次重试也失败，返回空字符串
                self._calls += 1
                self._total_time += time.time() - t0
                return f"[API_ERROR: {msg}]"

    def _call_api(self, question: str, context: str, png_path: str,
                  mode: str) -> str:
        import requests

        content_parts = []

        if png_path and os.path.exists(png_path) and mode in ("scan", "hybrid"):
            with open(png_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        system_prompt = (
            "你是一个记忆图分析助手。请根据提供的记忆图回答用户的问题。"
            "记忆图包含话题节点（颜色代表时间远近、大小代表重要性、"
            "红色粗边框+星标代表用户圈注的重点）、关系边（实线=因果、虚线=引用、"
            "红色双向=矛盾、点线=延伸）、以及话题旁注的评语。"
        )

        if mode == "scan":
            prompt = f"{question}\n\n(请仅根据上面这张记忆图来回答，不需要其他信息。)"
        elif mode == "hybrid":
            prompt = (
                f"{question}\n\n请先扫描记忆图定位相关信息，"
                f"然后参考以下对话文本来精确回答:\n\n{context[:800]}"
            )
        else:
            prompt = f"{question}\n\n请根据以下对话内容来回答:\n\n{context}"
            content_parts = []  # read 模式不发图

        content_parts.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": 0.3, "max_tokens": 2048},
            timeout=300,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, question: str, context: str, png_path: str,
                     mode: str) -> str:
        try:
            import ollama
        except ImportError:
            return self._call_mock(question, context, mode)

        images = []
        if png_path and os.path.exists(png_path) and mode in ("scan", "hybrid"):
            images.append(png_path)

        if mode == "scan":
            prompt = question + "\n(仅根据记忆图回答)"
        elif mode == "hybrid":
            prompt = question + f"\n(参考记忆图和以下文本: {context[:800]})"
        else:
            prompt = question + f"\n(文本内容: {context})"

        resp = ollama.chat(
            model=os.environ.get("OLLAMA_MODEL", "llava:7b"),
            messages=[{"role": "user", "content": prompt, "images": images}],
        )
        return resp["message"]["content"]

    def _call_mock(self, question: str, context: str, mode: str) -> str:
        """离线模拟回答 (仅用于管道测试)。"""
        q_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', question.lower()))
        if mode == "scan":
            ctx = context[:len(context)//2]
        elif mode == "hybrid":
            ctx = context[:len(context)*2//3]
        else:
            ctx = context
        relevant_lines = []
        for line in ctx.split("\n"):
            line_words = set(re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}', line.lower()))
            if line_words & q_words:
                relevant_lines.append(line.strip())
        return " ".join(relevant_lines[:8])

    @property
    def stats(self) -> dict:
        return {
            "backend": self.backend,
            "calls": self._calls,
            "total_time_s": round(self._total_time, 1),
            "avg_time_s": round(self._total_time / max(1, self._calls), 2),
        }


# 全局实例
_evaluator: EvaluatorAPI = None


def get_evaluator(force_mode: str = None) -> EvaluatorAPI:
    global _evaluator
    if _evaluator is None or force_mode:
        _evaluator = EvaluatorAPI(backend=force_mode if force_mode not in (None, "", "offline") else None)
        if force_mode == "offline":
            _evaluator.backend = "mock"
    return _evaluator


def ask_model(question: str, context: str, mode: str,
              png_path: str = None) -> str:
    """统一模型问答入口"""
    ev = get_evaluator()
    return ev.ask_with_graph(question, context, png_path, mode)


def _find_graph_png(data_path: str, sample: dict = None) -> str:
    """查找或渲染与数据文件关联的记忆图 PNG。

    优先级:
      1. 同目录下的 memory_graph_v*.png
      2. 从 sample['memory_graph'] dict 临时渲染 (仅 api/ollama 模式)
      3. mock 模式返回 None (跳过渲染)
    """
    ev = get_evaluator()
    if ev.backend == "mock":
        return None

    data_dir = os.path.dirname(data_path)
    png_files = sorted(Path(data_dir).glob("memory_graph_v*.png"))
    if png_files:
        return str(png_files[0])

    if sample and sample.get("memory_graph"):
        return _render_graph_temp(sample["memory_graph"])
    return None


_temp_png_counter = 0
_temp_png_cache = {}


def _render_graph_temp(graph: dict) -> str:
    """将记忆图 dict 渲染为临时 PNG，返回路径。有缓存避免重复渲染。"""
    global _temp_png_counter
    graph_id = id(graph)
    if graph_id in _temp_png_cache:
        return _temp_png_cache[graph_id]

    _temp_png_counter += 1
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".temp_eval_pngs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"temp_graph_{_temp_png_counter}.png")

    try:
        from phase3.render_memory_graph import render_memory_graph
        render_memory_graph(graph, output_path=out_path)
        _temp_png_cache[graph_id] = out_path
        return out_path
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  Exp2 评估: 增量记忆保持率
# ═══════════════════════════════════════════════════════════════

def eval_exp2(data_path: str, mode: str = "offline"):
    """评估 Exp2: 各版本下旧信息保留率"""
    data = load_json(data_path)
    results = {
        "experiment": "Exp2_IncrementalMemory",
        "mode": mode,
        "num_sessions": data.get("num_sessions", 0),
        "version_results": [],
    }

    memory_versions = data.get("memory_versions", [])

    for v in memory_versions:
        graph = v.get("memory_graph", {})
        nodes = graph.get("nodes", [])
        old_nodes = [n for n in nodes if n.get("time_factor", 0) < 0.4]
        new_nodes = [n for n in nodes if n.get("time_factor", 0) > 0.6]
        mid_nodes = [n for n in nodes if 0.4 <= n.get("time_factor", 0) <= 0.6]

        results["version_results"].append({
            "version": v["version"],
            "total_rounds": v.get("total_rounds", 0),
            "total_nodes": len(nodes),
            "old_nodes_retained": len(old_nodes),
            "new_nodes": len(new_nodes),
            "mid_nodes": len(mid_nodes),
            "retention_ratio": round(len(old_nodes) / max(1, len(nodes)), 4),
            "old_node_labels": [n["label"] for n in old_nodes[:5]],
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  Exp3 评估: 标注效果消融
# ═══════════════════════════════════════════════════════════════

def eval_exp3(data_path: str, mode: str = "offline"):
    """评估 Exp3: 4版图对比"""
    data = load_json(data_path)
    ev = get_evaluator(mode)
    results = {
        "experiment": "Exp3_Ablation",
        "mode": ev.backend,
        "total_samples": len(data),
        "graph_versions": {
            "A_skeleton": {"scores": [], "correct": 0, "total": 0},
            "B_time_color": {"scores": [], "correct": 0, "total": 0},
            "C_circle_marker": {"scores": [], "correct": 0, "total": 0},
            "D_full_memory": {"scores": [], "correct": 0, "total": 0},
        },
        "by_question_type": defaultdict(lambda: defaultdict(list)),
    }

    graph_png_cache = {}

    for sample_idx, sample in enumerate(data):
        graphs = sample.get("graphs", {})
        for qa in sample.get("qa_pairs", []):
            gt = qa.get("answer", "")
            question = qa.get("question", "")
            qtype = qa.get("type", "fact")

            for gv_name in ["A_skeleton", "B_time_color", "C_circle_marker", "D_full_memory"]:
                if gv_name not in graphs:
                    continue
                gv_text = _graph_to_text(graphs[gv_name])
                cache_key = f"{sample_idx}_{gv_name}"
                if cache_key not in graph_png_cache:
                    graph_png_cache[cache_key] = _render_graph_temp(graphs[gv_name])
                png_path = graph_png_cache[cache_key]

                pred = ask_model(question, gv_text, "scan", png_path)
                score = simple_match_score(pred, gt)

                results["graph_versions"][gv_name]["scores"].append(score)
                results["graph_versions"][gv_name]["total"] += 1
                if score > 0.3:
                    results["graph_versions"][gv_name]["correct"] += 1

                results["by_question_type"][qtype][gv_name].append(score)

    for gv in results["graph_versions"]:
        scores = results["graph_versions"][gv]["scores"]
        total = results["graph_versions"][gv]["total"]
        results["graph_versions"][gv]["mean_score"] = round(sum(scores) / len(scores), 4) if scores else 0
        results["graph_versions"][gv]["accuracy"] = round(
            results["graph_versions"][gv]["correct"] / total, 4) if total > 0 else 0

    for qtype in results["by_question_type"]:
        for gv in list(results["by_question_type"][qtype].keys()):
            scores = results["by_question_type"][qtype][gv]
            results["by_question_type"][qtype][f"{gv}_mean"] = round(sum(scores) / len(scores), 4) if scores else 0

    return results


# ═══════════════════════════════════════════════════════════════
#  Exp4 评估: 规模扩展曲线
# ═══════════════════════════════════════════════════════════════

def eval_exp4(data_path: str, mode: str = "offline"):
    """评估 Exp4: 不同规模下的优势比"""
    data = load_json(data_path)
    ev = get_evaluator(mode)

    if isinstance(data, list):
        all_data = data
    else:
        all_data = [data]

    results = {
        "experiment": "Exp4_Scale",
        "mode": ev.backend,
        "scale_results": [],
    }

    png_cache = {}

    for size_idx, size_data in enumerate(all_data):
        size_label = size_data.get("size", "?")
        num_rounds = size_data.get("num_rounds", 0)
        conv_text = _conversation_to_text(size_data.get("conversation", []))
        graph = size_data.get("memory_graph", {})
        graph_text = _graph_to_text(graph)

        cache_key = f"scale_{size_label}"
        if cache_key not in png_cache:
            png_cache[cache_key] = _render_graph_temp(graph)
        png_path = png_cache[cache_key]

        conv_tokens = count_tokens(conv_text)
        graph_tokens = count_tokens(graph_text)

        scores_graph = []
        scores_text = []
        scores_hybrid = []

        for qa in size_data.get("qa_pairs", []):
            gt = qa.get("answer", "")
            question = qa.get("question", "")
            pred_g = ask_model(question, graph_text, "scan", png_path)
            pred_t = ask_model(question, conv_text, "read")
            pred_h = ask_model(question, graph_text + "\n" + conv_text[:conv_tokens//3], "hybrid", png_path)
            scores_graph.append(simple_match_score(pred_g, gt))
            scores_text.append(simple_match_score(pred_t, gt))
            scores_hybrid.append(simple_match_score(pred_h, gt))

        mean_g = sum(scores_graph) / len(scores_graph) if scores_graph else 0
        mean_t = sum(scores_text) / len(scores_text) if scores_text else 0
        mean_h = sum(scores_hybrid) / len(scores_hybrid) if scores_hybrid else 0

        results["scale_results"].append({
            "size": size_label,
            "num_rounds": num_rounds,
            "text_tokens": conv_tokens,
            "graph_tokens": graph_tokens,
            "token_ratio": round(graph_tokens / max(1, conv_tokens), 4),
            "graph_accuracy": round(mean_g, 4),
            "text_accuracy": round(mean_t, 4),
            "hybrid_accuracy": round(mean_h, 4),
            "advantage_ratio": round(mean_g / max(0.001, mean_t), 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════
#  Exp5 评估: 训练后主动生成率
# ═══════════════════════════════════════════════════════════════

def eval_exp5(data_path: str, mode: str = "offline"):
    """评估 Exp5: 检查训练数据中记忆图生成质量"""
    data = load_json(data_path)
    results = {
        "experiment": "Exp5_TrainingData",
        "total_samples": len(data),
        "has_mermaid": 0,
        "has_summary": 0,
        "domains": defaultdict(int),
    }

    for sample in data:
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "<memory_sketch>" in content:
                    results["has_mermaid"] += 1
                if "<summary>" in content:
                    results["has_summary"] += 1
        results["domains"][sample.get("domain", "unknown")] += 1

    results["mermaid_rate"] = round(results["has_mermaid"] / max(1, len(data)), 4)
    results["summary_rate"] = round(results["has_summary"] / max(1, len(data)), 4)

    return results


# ═══════════════════════════════════════════════════════════════
#  Exp6 评估: 零样本转移
# ═══════════════════════════════════════════════════════════════

def eval_exp6(data_path: str, mode: str = "offline"):
    """评估 Exp6: 零样本转移测试"""
    data = load_json(data_path)
    tasks = defaultdict(int)

    for sample in data:
        tasks[sample.get("task", "unknown")] += 1

    results = {
        "experiment": "Exp6_Transfer",
        "total_samples": len(data),
        "task_distribution": dict(tasks),
    }

    return results


# ═══════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 图形记忆系统评估")
    parser.add_argument("--all", action="store_true", help="运行全部评估")
    parser.add_argument("--exp1", action="store_true")
    parser.add_argument("--exp2", action="store_true")
    parser.add_argument("--exp3", action="store_true")
    parser.add_argument("--exp4", action="store_true")
    parser.add_argument("--exp5", action="store_true")
    parser.add_argument("--exp6", action="store_true")
    parser.add_argument("--data", "-d", default="./phase3_data/", help="数据目录")
    parser.add_argument("--output", "-o", default="./phase3_eval_results/", help="结果输出目录")
    parser.add_argument("--mode", "-m", choices=["offline", "api", "ollama"], default="offline",
                        help="评估模式: offline=关键词模拟, api=云端多模态API, ollama=本地Ollama")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)

    evals = {
        "exp1": (args.exp1, "exp1_static/exp1_static_memory.json", eval_exp1),
        "exp2": (args.exp2, "exp2_incremental/exp2_incremental_memory.json", eval_exp2),
        "exp3": (args.exp3, "exp3_ablation/exp3_ablation.json", eval_exp3),
        "exp4": (args.exp4, "exp4_scale/exp4_scale_all.json", eval_exp4),
        "exp5": (args.exp5, "exp5_train/exp5_train_data.json", eval_exp5),
        "exp6": (args.exp6, "exp6_transfer/exp6_transfer.json", eval_exp6),
    }

    run_all = args.all
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    full_report = {
        "phase": "Phase 3 — Graphical Memory System Evaluation",
        "timestamp": timestamp,
        "mode": args.mode,
        "experiments": {},
    }

    eval_mode = None if args.mode == "offline" else args.mode
    ev = get_evaluator(eval_mode)
    print(f"评估后端: {ev.backend}")

    for exp_name, (should_run, rel_path, eval_func) in evals.items():
        if not run_all and not should_run:
            continue

        data_path = data_dir / rel_path
        if not data_path.exists():
            print(f"跳过 {exp_name}: 数据文件不存在 {data_path}")
            continue

        print(f"运行 {exp_name}...")
        try:
            result = eval_func(str(data_path), args.mode)
            full_report["experiments"][exp_name] = result

            exp_out = out_dir / f"{exp_name}_{timestamp}.json"
            save_json(result, str(exp_out))
            print(f"  → {exp_out}")
        except Exception as e:
            print(f"  [FAIL] {exp_name}: {e}")

    report_path = out_dir / f"phase3_report_{timestamp}.json"
    save_json(full_report, str(report_path))
    print(f"\n完整报告: {report_path}")
    print(f"API调用统计: {ev.stats}")
    _print_summary(full_report)


def _print_summary(report: dict):
    """打印评估摘要"""
    print("\n" + "=" * 60)
    print("Phase 3 图形记忆系统 — 评估摘要")
    print("=" * 60)

    for exp_name, exp_data in report.get("experiments", {}).items():
        exp_label = exp_data.get("experiment", exp_name)
        print(f"\n### {exp_label}")

        if exp_name == "exp1":
            conds = exp_data.get("conditions", {})
            for cname, cdata in conds.items():
                print(f"  {cname}: mean={cdata.get('mean_score', 0):.3f}, "
                      f"acc={cdata.get('accuracy', 0):.3f}")
            tc = exp_data.get("token_comparison", {})
            print(f"  Token: graph={tc.get('graph_avg_tokens', 0)}, "
                  f"text={tc.get('text_avg_tokens', 0)}")

        elif exp_name == "exp2":
            for vr in exp_data.get("version_results", []):
                print(f"  v{vr['version']}: {vr['total_nodes']} nodes, "
                      f"retention={vr['retention_ratio']:.3f} "
                      f"(old={vr['old_nodes_retained']}, new={vr['new_nodes']})")

        elif exp_name == "exp3":
            gvs = exp_data.get("graph_versions", {})
            for gv_name, gv_data in gvs.items():
                print(f"  {gv_name}: mean={gv_data.get('mean_score', 0):.3f}, "
                      f"acc={gv_data.get('accuracy', 0):.3f}")

        elif exp_name == "exp4":
            for sr in exp_data.get("scale_results", []):
                print(f"  {sr['size']} ({sr['num_rounds']}轮): "
                      f"graph={sr['graph_accuracy']:.3f}, "
                      f"text={sr['text_accuracy']:.3f}, "
                      f"ratio={sr['advantage_ratio']:.3f}, "
                      f"token_ratio={sr['token_ratio']:.3f}")

        elif exp_name == "exp5":
            print(f"  样本数: {exp_data.get('total_samples', 0)}, "
                  f"mermaid_rate={exp_data.get('mermaid_rate', 0):.1%}")

        elif exp_name == "exp6":
            print(f"  样本数: {exp_data.get('total_samples', 0)}, "
                  f"任务分布: {exp_data.get('task_distribution', {})}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
