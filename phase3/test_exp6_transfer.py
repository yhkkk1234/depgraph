#!/usr/bin/env python
"""
test_exp6_transfer.py — Phase 3b-2: 零样本转移测试
测试微调后的模型在未见过的任务类型上是否自发产生图式记忆行为

用法:
  python test_exp6_transfer.py \
    --lora phase3_output/training/phase3_lora \
    --data phase3_output/data/exp6_transfer/exp6_transfer.json \
    --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import fire, json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def test_transfer(
    lora: str = "phase3_output/training/phase3_lora",
    data_path: str = "phase3_output/data/exp6_transfer/exp6_transfer.json",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 1024,
):
    """加载 LoRA 模型，测试零样本转移"""

    print(f"Model: {model_name}")
    print(f"LoRA: {lora}")
    print(f"Data: {data_path}")

    # ── Load data ──
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # ── Load model ──
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, lora)
    model.eval()
    print("Model loaded.\n")

    # ── Group by task ──
    tasks = {}
    for s in samples:
        t = s.get("task", "unknown")
        tasks.setdefault(t, []).append(s)

    results = {}

    for task_name, task_samples in tasks.items():
        task_label = task_samples[0].get("task_name", task_name)
        print(f"{'='*60}")
        print(f"Task: {task_label} ({task_name}) — {len(task_samples)} samples")
        print(f"{'='*60}")

        has_sketch = 0
        has_structure = 0
        has_graph_term = 0
        total = 0

        for si, sample in enumerate(task_samples[:3]):  # 每种任务测3个
            prompt = sample.get("prompt", "")
            if not prompt:
                continue
            total += 1

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.3, do_sample=True, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # 检测结构化行为
            h_sketch = any(kw in response for kw in [
                "<memory_sketch>", "```mermaid", "graph TD", "graph LR",
                "flowchart", "-->|", "--> ",
            ])
            h_structure = any(kw in response for kw in [
                "第一步", "第二步", "第一", "第二", "首先", "其次",
                "\n1.", "\n2.", "\n- ", "\n* ",
                "### ", "## 总结", "核心要点",
            ])
            h_graph = any(kw in response for kw in [
                "节点", "关系图", "拓扑", "依赖", "关联", "结构图",
                "箭头", "连线", "node", "edge", "graph",
            ])

            has_sketch += int(h_sketch)
            has_structure += int(h_structure)
            has_graph_term += int(h_graph)

            print(f"\n  Sample {si+1}:")
            print(f"    sketch={'YES' if h_sketch else 'no'}  "
                  f"structure={'YES' if h_structure else 'no'}  "
                  f"graph_term={'YES' if h_graph_term else 'no'}")
            print(f"    Response: {response[:200]}...")

        rate_sketch = has_sketch / max(1, total)
        rate_structure = has_structure / max(1, total)
        rate_graph = has_graph_term / max(1, total)

        print(f"\n  Task summary [{task_label}]:")
        print(f"    graph_generation:  {has_sketch}/{total} = {rate_sketch:.0%}")
        print(f"    structure:         {has_structure}/{total} = {rate_structure:.0%}")
        print(f"    graph_terms:       {has_graph_term}/{total} = {rate_graph:.0%}")

        results[task_name] = {
            "label": task_label,
            "samples_tested": total,
            "sketch_rate": round(rate_sketch, 2),
            "structure_rate": round(rate_structure, 2),
            "graph_term_rate": round(rate_graph, 2),
        }

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("EXP6: ZERO-SHOT TRANSFER — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  {'Task':<25s} {'Sketch':>8s} {'Struct':>8s} {'GraphTerm':>10s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")

    total_sketch, total_struct, total_graph, total_samples = 0, 0, 0, 0
    for tname, r in results.items():
        print(f"  {r['label']:<25s} {r['sketch_rate']:>7.0%}  {r['structure_rate']:>7.0%}  {r['graph_term_rate']:>9.0%}")
        total_sketch += int(r['sketch_rate'] * r['samples_tested'])
        total_struct += int(r['structure_rate'] * r['samples_tested'])
        total_graph += int(r['graph_term_rate'] * r['samples_tested'])
        total_samples += r['samples_tested']

    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    print(f"  {'OVERALL':<25s} {total_sketch/total_samples:>7.0%}  {total_struct/total_samples:>7.0%}  {total_graph/total_samples:>9.0%}")

    # 判定结果
    overall_sketch = total_sketch / max(1, total_samples)
    print(f"\n  Transfer verdict:")
    if overall_sketch > 0.3:
        print(f"  ✅ POSITIVE TRANSFER ({overall_sketch:.0%}): Model spontaneously produces graph sketches on unseen tasks!")
    elif overall_sketch > 0.1:
        print(f"  ⚠️ PARTIAL TRANSFER ({overall_sketch:.0%}): Some graph behavior transferring, more training needed.")
    else:
        print(f"  ❌ NO TRANSFER ({overall_sketch:.0%}): Graph behavior not internalized. Consistent with Phase 2 Exp4 (0%).")

    ts = __import__("time").strftime("%Y%m%d_%H%M%S")
    out_path = f"phase3_output/eval_results/exp6_transfer_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "results": results,
                    "overall_sketch_rate": round(overall_sketch, 2)}, f, ensure_ascii=False, indent=2)
    print(f"\nResult saved: {out_path}")


if __name__ == "__main__":
    fire.Fire(test_transfer)
