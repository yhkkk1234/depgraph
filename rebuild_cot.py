"""
Rebuild training data with structured diagram description prefix.
Adds CoT (Chain-of-Thought) diagram analysis to every assistant response.
"""
import json, os, sys, hashlib, base64
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from scan_deps import scan_project
from render_diagram import render_dependency_graph

SCRIPT = os.path.dirname(__file__)
OUT_DIR = os.path.join(SCRIPT, "training_data", "cot_format")
IMG_DIR = os.path.join(OUT_DIR, "images")
SRC = os.path.join(SCRIPT, "training_data", "merged_dataset.jsonl")
os.makedirs(IMG_DIR, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

converted = []
img_cache = {}  # hash -> filename

for i, s in enumerate(samples):
    meta = s.get("metadata", {})
    bug_mod = meta.get("bug_module", "unknown")
    bug_func = meta.get("bug_func", "unknown")
    change = meta.get("change_desc", "")
    affected = meta.get("affected_files", [])

    # Build structured diagram description from ground truth
    affected_list = "\n".join(f"  - `{f}`" for f in affected)
    diagram_desc = f"""## Diagram Analysis

The dependency diagram shows the module `{bug_mod}` highlighted in red, indicating it was changed.
Yellow-highlighted modules are direct dependents, connected via dependency arrows to `{bug_mod}`:

{affected_list}

Total: {len(affected)} modules directly depend on `{bug_mod}`. Each may call `{bug_func}()` and needs inspection.

## Impact Assessment & Fix

"""

    # Rewrite assistant message with diagram description prefix
    new_msgs = []
    new_images = []

    for msg in s["messages"]:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, str):
            # Prepend diagram description
            content = diagram_desc + content
            new_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multi-modal user message: extract images
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    h = hashlib.md5(url.encode()).hexdigest()[:12]
                    if h not in img_cache:
                        fn = f"img_{h}.png"
                        with open(os.path.join(IMG_DIR, fn), "wb") as fimg:
                            fimg.write(base64.b64decode(url))
                        img_cache[h] = fn
                    new_images.append(f"images/{img_cache[h]}")
                    text_parts.append("<image>")
            new_msgs.append({"role": role, "content": "\n".join(text_parts)})
        else:
            new_msgs.append({"role": role, "content": content})

    entry = {"messages": new_msgs, "images": new_images, "metadata": meta}
    converted.append(entry)

# Write output
out_jsonl = os.path.join(OUT_DIR, "dataset.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for e in converted:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

dataset_info = {
    "diagram_fix_cot": {
        "file_name": "dataset.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
}
with open(os.path.join(OUT_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2)

print(f"CoT dataset: {OUT_DIR}")
print(f"  Samples: {len(converted)}")
print(f"  Images:  {len(os.listdir(IMG_DIR))}")
print(f"  Size:    {os.path.getsize(out_jsonl)/1024/1024:.1f} MB")
print("\nUpload to: /root/autodl-tmp/cot_format/")
