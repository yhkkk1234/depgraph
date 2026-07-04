"""
Rebuild training data with DUAL input: diagram image + text legend.
No OCR needed — model reads names from text, understands topology from image.
"""
import json, os, sys, hashlib, base64
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from legend_gen import generate_legend, generate_full_legend

SCRIPT = os.path.dirname(__file__)
OUT_DIR = os.path.join(SCRIPT, "training_data", "dual_format")
IMG_DIR = os.path.join(OUT_DIR, "images")
SRC = os.path.join(SCRIPT, "training_data", "merged_dataset.jsonl")
os.makedirs(IMG_DIR, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

# We need project-specific dependency graphs to generate legends.
# Map project names to their paths. Use known projects from site-packages.
def get_project_path(proj_name):
    import site, importlib
    for site_dir in site.getsitepackages():
        proj_dir = os.path.join(site_dir, proj_name)
        if os.path.isdir(proj_dir):
            return proj_dir
    return None

# Pre-scan projects
from scan_deps import scan_project
legend_cache = {}

def get_legend(proj_name, bug_module):
    key = (proj_name, bug_module)
    if key in legend_cache:
        return legend_cache[key]
    proj_path = get_project_path(proj_name)
    if proj_path:
        try:
            g = scan_project(proj_path)
            legend = generate_full_legend(g)
            legend_cache[key] = legend
            return legend
        except:
            pass
    return None

converted = 0
img_cache = {}
for s in samples:
    meta = s.get("metadata", {})
    proj_name = meta.get("project", s.get("project", "unknown"))
    bug_mod = meta.get("bug_module", "unknown")
    bug_func = meta.get("bug_func", "unknown")
    change = meta.get("change_desc", "")
    affected = meta.get("affected_files", [])

    # Build legend text
    legend_text = get_legend(proj_name, bug_mod) or ""
    affected_list = "\n".join(f"  - `{f}`" for f in affected)

    # Diagram description + legend preamble for assistant
    diagram_preamble = f"""## Diagram Analysis

The dependency diagram shows `{bug_mod}` (red, changed). Yellow modules are direct dependents.

Text legend confirms:
{affected_list}

Total: {len(affected)} modules directly depend on `{bug_mod}`.

## Impact Assessment & Fix

"""

    # Rewrite messages
    new_msgs = []
    new_images = []
    for msg in s["messages"]:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user" and isinstance(content, list):
            # Multi-modal user message: add legend before the image
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    t = part["text"]
                    # Insert legend text before the image reference
                    if "<image>" in t and legend_text:
                        t = t.replace("<image>", f"\n{legend_text}\n\nDIAGRAM:\n<image>")
                    text_parts.append(t)
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    h = hashlib.md5(url.encode()).hexdigest()[:12]
                    if h not in img_cache:
                        fn = f"img_{h}.png"
                        with open(os.path.join(IMG_DIR, fn), "wb") as fimg:
                            fimg.write(base64.b64decode(url))
                        img_cache[h] = fn
                    new_images.append(f"images/{img_cache[h]}")
            new_msgs.append({"role": role, "content": "\n".join(text_parts)})
        elif role == "assistant" and isinstance(content, str):
            new_msgs.append({"role": role, "content": diagram_preamble + content})
        else:
            new_msgs.append(msg)

    entry = {"messages": new_msgs, "images": new_images, "metadata": meta}
    converted += 1

    # Write incrementally
    out_jsonl = os.path.join(OUT_DIR, "dataset.jsonl")
    with open(out_jsonl, "a" if converted > 1 else "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# dataset_info.json
dataset_info = {
    "diagram_fix_dual": {
        "file_name": "dataset.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
}
with open(os.path.join(OUT_DIR, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2)

print(f"Dual-format dataset: {OUT_DIR}")
print(f"  Samples: {converted}")
print(f"  Images:  {len(os.listdir(IMG_DIR))}")
print(f"Upload to: /root/autodl-tmp/dual_format/")
