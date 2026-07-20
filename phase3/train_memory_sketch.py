#!/usr/bin/env python
"""
train_memory_sketch.py — Phase 3b-1 QLoRA Training Script

训练模型在对话后主动生成 <memory_sketch> Mermaid 记忆图 + <summary> 总结

兼容 Python 3.8+, PyTorch 2.0+, 直接使用 transformers+peft 而非 LLaMA-Factory
"""
from __future__ import annotations

import fire
import json, os, torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


class MemorySketchDataset(Dataset):
    """ShareGPT 格式数据集 → 训练用 tokenized 数据"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            conversations = item.get("conversations", [])
            if len(conversations) < 2:
                continue

            human_msgs = [c["value"] for c in conversations if c.get("from") == "human"]
            gpt_msgs = [c["value"] for c in conversations if c.get("from") == "gpt"]

            if not human_msgs or not gpt_msgs:
                continue

            user_text = "\n".join(human_msgs)
            assistant_text = "\n".join(gpt_msgs)

            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            self.samples.append(text)

        print(f"  Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokenized = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        # 因果 LM: labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokenized.items()}


def train(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    train_data: str = "phase3_output/training/phase3_memory/phase3_memory_train.json",
    val_data: str = "phase3_output/training/phase3_memory/phase3_memory_val.json",
    output_dir: str = "phase3_output/training/phase3_lora",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 5,
    max_length: int = 2048,
    save_steps: int = 100,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf8 = True

    print(f"\n{'='*50}")
    print(f"Training Config:")
    print(f"  Model: {model_name}")
    print(f"  LoRA: r={lora_rank}, alpha={lora_alpha}")
    print(f"  Batch: {batch_size}, GradAcc: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"  Max length: {max_length}")
    print(f"{'='*50}\n")

    # ── Tokenizer ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──
    print("Loading datasets...")
    train_dataset = MemorySketchDataset(train_data, tokenizer, max_length)
    val_dataset = MemorySketchDataset(val_data, tokenizer, max_length)

    # ── Model with 4-bit QLoRA ──
    print("Loading model (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Trainer ──
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        dataloader_num_workers=0,
        report_to="none",
        evaluation_strategy="steps",
        save_strategy="steps",
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nTraining...")
    trainer.train()

    print(f"\nSaving LoRA weights to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nDone!")


if __name__ == "__main__":
    fire.Fire(train)
