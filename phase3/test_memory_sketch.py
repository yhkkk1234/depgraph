#!/usr/bin/env python
"""
test_memory_sketch.py — 测试微调后的模型是否学会了主动生成记忆图

用法:
  python test_memory_sketch.py \
    --lora phase3_output/training/phase3_lora \
    --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import fire, json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# 测试对话样本
TEST_CONVERSATIONS = [
    {
        "domain": "软件架构",
        "conversation": """以下是一段对话记录。请分析对话内容，先画出记忆图（Mermaid 格式）梳理话题结构，标记重点，然后总结核心要点。

## 对话记录

[第1轮] 用户: 关于微服务拆分，我有些想法。这个很重要，需要重点跟进
[第2轮] 助手: 好的，微服务拆分确实是个需要仔细考虑的点。从整体架构来看，它和哪些模块有耦合？
[第3轮] 用户: 具体来说，API网关和服务发现这块需要优先处理。先记下来，回头再评估
[第4轮] 助手: 收到。API网关这块我可以先从几个角度帮你梳理一下。
[第5轮] 用户: 另外数据库拆分也需要考虑，特别是跨服务事务的问题。这是核心问题，必须解决
[第6轮] 助手: 明白，跨服务事务确实是微服务的核心挑战。我记下了。
[第7轮] 用户: 缓存策略也需要调整，之前单体的缓存方案不适用了。参考一下业界最佳实践
[第8轮] 助手: 关于缓存策略，能不能展开说说具体是哪个方面让你觉得有问题？

请输出记忆图和总结。""",
    },
    {
        "domain": "团队管理",
        "conversation": """以下是一段对话记录。请分析对话内容，先画出记忆图（Mermaid 格式）梳理话题结构，标记重点，然后总结核心要点。

## 对话记录

[第1轮] 用户: 关于代码审查流程，我最近有些想法。现在review太慢了，影响交付速度
[第2轮] 助手: 好的，代码审查流程这个方向我之前也关注过，有几个关键点要注意。
[第3轮] 用户: 我觉得核心问题是reviewer太分散，没人专职做。需要和其他团队对齐
[第4轮] 助手: 了解。reviewer分配这个问题我可以先帮你梳理下现有的工作量。
[第5轮] 用户: 另外技术债务管理也需要抓起来，最近线上问题很多是历史欠账。这是核心问题，必须解决
[第6轮] 助手: 明白，这个我记下来。技术债务的优先级你觉得应该怎么定？
[第7轮] 用户: 可以先从高频变更模块入手，把最危险的先修了
[第8轮] 助手: 好的，高频模块优先修复，这个策略很务实。

请输出记忆图和总结。""",
    },
]


def test(
    lora: str = "phase3_output/training/phase3_lora",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 1024,
):
    """加载 LoRA 模型并测试记忆图生成能力"""

    print(f"Loading model: {model_name}")
    print(f"LoRA weights: {lora}")

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
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if os.path.exists(lora):
        model = PeftModel.from_pretrained(base_model, lora)
        print("LoRA adapter loaded!")
    else:
        print(f"WARNING: LoRA not found at {lora}, using base model")
        model = base_model

    model.eval()

    for i, sample in enumerate(TEST_CONVERSATIONS):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {sample['domain']}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": sample["conversation"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        has_sketch = "<memory_sketch>" in response or "graph " in response.lower() or "```mermaid" in response
        has_summary = "<summary>" in response or "总结" in response or "核心要点" in response
        has_structure = "-->" in response or "-->|" in response

        print(f"  memory_sketch: {'YES' if has_sketch else 'NO'}")
        print(f"  summary: {'YES' if has_summary else 'NO'}")
        print(f"  graph arrows: {'YES' if has_structure else 'NO'}")
        print(f"\n{response[:800]}")
        if len(response) > 800:
            print(f"... ({len(response)} chars total)")

    print(f"\n{'='*60}")
    print("Test complete!")


if __name__ == "__main__":
    fire.Fire(test)
