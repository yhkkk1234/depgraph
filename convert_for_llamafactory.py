"""
Convert merged_dataset.jsonl to LLaMA-Factory format.
Extracts embedded base64 images to separate PNG files,
replaces inline image_url with <image> tag + image path reference.
"""
import json
import hashlib
import base64
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(__file__)
TDIR = os.path.join(SCRIPT_DIR, "training_data")
SRC = os.path.join(TDIR, "merged_dataset.jsonl")
OUT_DIR = os.path.join(TDIR, "llamafactory_format")
IMG_DIR = os.path.join(OUT_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

converted = []
image_hashes = {}  # hash -> filename, deduplicate images

for i, sample in enumerate(samples):
    new_msgs = []
    image_paths = []

    for msg in sample["messages"]:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, list):
            # Multi-modal message: extract image, build text
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    # Generate image filename from hash
                    img_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                    if img_hash not in image_hashes:
                        fname = f"img_{img_hash}.png"
                        fpath = os.path.join(IMG_DIR, fname)
                        with open(fpath, "wb") as fimg:
                            fimg.write(base64.b64decode(url))
                        image_hashes[img_hash] = fname
                    image_paths.append(f"images/{image_hashes[img_hash]}")
                    text_parts.append("<image>")

            # LLaMA-Factory expects text with <image> tag
            new_msgs.append({
                "role": role,
                "content": "\n".join(text_parts),
            })
        elif isinstance(content, str):
            # Pure text message
            new_msgs.append({
                "role": role,
                "content": content,
            })

    entry = {
        "messages": new_msgs,
        "images": image_paths,
    }

    # Preserve metadata for quality filtering
    if "metadata" in sample:
        entry["metadata"] = sample["metadata"]

    converted.append(entry)

# Write converted dataset
out_jsonl = os.path.join(OUT_DIR, "dataset.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for entry in converted:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Write dataset_info.json for LLaMA-Factory
dataset_info = {
    "diagram_fix": {
        "file_name": "dataset.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
}

with open(os.path.join(OUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2)

# Stats
img_count = len(os.listdir(IMG_DIR))
perfect = sum(1 for e in converted if e.get("metadata", {}).get("score", {}).get("quality") == "perfect")
good = sum(1 for e in converted if e.get("metadata", {}).get("score", {}).get("quality") == "good")

print(f"Converted: {OUT_DIR}")
print(f"  dataset.jsonl: {len(converted)} samples")
print(f"  images/: {img_count} unique images")
print(f"  Quality: {perfect} perfect, {good} good")
print(f"  dataset_info.json: ready for LLaMA-Factory")
print(f"\nUpload this directory to AutoDL: {OUT_DIR}")
