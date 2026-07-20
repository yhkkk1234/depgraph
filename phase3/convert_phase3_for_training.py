#!/usr/bin/env python
"""
convert_phase3_for_training.py — 将 Exp5 训练数据转换为 LLaMA-Factory ShareGPT 格式

Phase 3b-1: 训练模型在对话后主动生成 memory_sketch + summary
数据格式: JSON → ShareGPT conversation 格式

用法:
  python convert_phase3_for_training.py \
    --input phase3_output/data/exp5_train/exp5_train_data.json \
    --output phase3_output/training/phase3_memory/
"""
from __future__ import annotations

import argparse, json, os, random
from pathlib import Path

random.seed(42)


def convert_to_sharegpt(input_path: str, output_dir: str,
                         train_ratio: float = 0.85):
    """转换 Exp5 训练数据为 LLaMA-Factory ShareGPT 格式"""

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = []
    for item in data:
        messages = item.get("messages", [])
        if len(messages) < 2:
            continue

        conv = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conv.append({"from": "human", "value": content})
            elif role == "assistant":
                conv.append({"from": "gpt", "value": content})

        if conv:
            conversations.append({"conversations": conv})

    random.shuffle(conversations)

    split_idx = int(len(conversations) * train_ratio)
    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]

    train_path = out_dir / "phase3_memory_train.json"
    val_path = out_dir / "phase3_memory_val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(out_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "phase3_memory_train": {
                "file_name": "phase3_memory_train.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
            },
            "phase3_memory_val": {
                "file_name": "phase3_memory_val.json",
                "formatting": "sharegpt",
                "columns": {"messages": "conversations"},
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(data)} samples → {len(train_data)} train + {len(val_data)} val")
    print(f"Output: {out_dir}")
    print(f"  {train_path} ({len(train_data)} samples)")
    print(f"  {val_path} ({len(val_data)} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    convert_to_sharegpt(args.input, args.output)
