#!/bin/bash
# Test: Does Qwen3-VL spontaneously call depgraph tool?
MODEL_PATH=$(find /root/autodl-tmp/models -name "config.json" -path "*Qwen3-VL*" | head -1 | xargs dirname)

python << 'PYEOF'
import torch, json, os
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = os.popen("find /root/autodl-tmp/models -name config.json -path '*Qwen3-VL*' | head -1 | xargs dirname").read().strip()
print(f"Model: {MODEL_PATH}")

model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Simulate: model calls depgraph → we return dependency data → model answers
depgraph_response = json.dumps({
    "changed": "models.task",
    "dependents": ["services/task_service.py", "services/notification.py", "api/handlers.py", "utils/validators.py"],
})

messages = [
    {"role": "system", "content": "You have a tool called depgraph(project, module). It returns which files depend on the given module. YOU MUST call it for cross-module bugs. Format: <tool_call>{\"name\": \"depgraph\", \"arguments\": {...}}</tool_call>"},
    {"role": "user", "content": "ERROR: Task.to_dict() signature changed in models/task.py. Which files are affected?"},
    # We force the tool call to simulate spontaneous behavior
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"depgraph\", \"arguments\": {\"project\": \"toy\", \"module\": \"models.task\"}}\n</tool_call>"},
    {"role": "tool", "content": depgraph_response},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512)
resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n=== Test: Tool → Response ===")
print(f"Response: {resp[:600]}...")

# Check if tool response data was used
expected = ["services/task_service.py", "services/notification.py", "api/handlers.py", "utils/validators.py"]
found = [f for f in expected if f.lower() in resp.lower()]
print(f"\nFiles used from tool response: {len(found)}/4 => {found}")
PYEOF
