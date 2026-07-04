"""
Document structure scanner: parse Markdown and LaTeX documents into
section dependency graphs. Tracks internal references between sections.
"""
import re, os
from pathlib import Path
from collections import defaultdict


def _scan_markdown(file_path: str) -> dict:
    """Parse Markdown headings and internal links."""
    text = Path(file_path).read_text(encoding="utf-8")
    sections = {}
    refs = defaultdict(set)  # source_section_id -> set of target_section_ids

    # Extract headings with auto-generated IDs
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = []
    for m in heading_pattern.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        # Generate slug ID (matches GitHub/Markdown convention)
        slug = title.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        headings.append({"level": level, "title": title, "id": slug, "pos": m.start()})

    # Build section hierarchy
    for i, h in enumerate(headings):
        # Find section content (from this heading to next heading of same or higher level)
        content_start = h["pos"]
        content_end = len(text)
        for j in range(i + 1, len(headings)):
            if headings[j]["level"] <= h["level"]:
                content_end = headings[j]["pos"]
                break
        h["content_start"] = content_start
        h["content_end"] = content_end
        sections[h["id"]] = {
            "title": h["title"],
            "level": h["level"],
            "dependencies": [],
            "dependents": [],
        }

    # Find cross-references between sections
    for src_h in headings:
        src_section = sections[src_h["id"]]
        src_content = text[src_h["content_start"]:src_h["content_end"]]

        for tgt_h in headings:
            if tgt_h["id"] == src_h["id"]:
                continue
            # Check if target section is referenced in source's content
            # Match patterns: [text](#section-id), (#section-id), "Section Title"
            if f"(#{tgt_h['id']})" in src_content or f"(#{tgt_h['id']}" in src_content:
                src_section["dependencies"].append(tgt_h["id"])
                refs[src_h["id"]].add(tgt_h["id"])
            # Also check for title reference
            if tgt_h["title"].lower() in src_content.lower():
                src_section["dependencies"].append(tgt_h["id"])
                refs[src_h["id"]].add(tgt_h["id"])

    return {
        "modules": sections,
        "refs": {k: list(v) for k, v in refs.items()},
        "type": "markdown",
    }


def _scan_latex(file_path: str) -> dict:
    """Parse LaTeX section commands and \\ref/\\cite references."""
    text = Path(file_path).read_text(encoding="utf-8")
    sections = {}
    refs = defaultdict(set)

    # Extract sections and labels
    section_pattern = re.compile(
        r"\\(section|subsection|subsubsection|chapter)\{([^}]+)\}", re.MULTILINE
    )
    label_pattern = re.compile(r"\\label\{([^}]+)\}")
    ref_pattern = re.compile(r"\\(ref|cite|eqref)\{([^}]+)\}")

    # Collect all labels and their positions
    labels = {}
    for m in label_pattern.finditer(text):
        labels[m.group(1)] = m.start()

    # Collect all \ref targets (labels)
    for m in ref_pattern.finditer(text):
        label = m.group(2)
        if label in labels:
            refs[labels[label]] = label  # Simple marker

    # Build sections
    sec_matches = list(section_pattern.finditer(text))
    for i, m in enumerate(sec_matches):
        sec_type = m.group(1)
        sec_title = m.group(2).strip()
        slug = re.sub(r"[^\w]+", "-", sec_title.lower())
        start = m.start()
        end = sec_matches[i + 1].start() if i + 1 < len(sec_matches) else len(text)

        # Find refs to other labels in this section's content
        section_deps = []
        section_text = text[start:end]
        for ref_m in ref_pattern.finditer(section_text):
            label = ref_m.group(2)
            if label in labels:
                # Find which section contains this label
                for j, sm in enumerate(sec_matches):
                    if labels[label] >= sm.start() and (j + 1 >= len(sec_matches) or labels[label] < sec_matches[j + 1].start()):
                        tgt_title = sm.group(2).strip()
                        tgt_slug = re.sub(r"[^\w]+", "-", tgt_title.lower())
                        if tgt_slug != slug:
                            section_deps.append(tgt_slug)

        sections[slug] = {
            "title": sec_title,
            "level": {"chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4}.get(sec_type, 2),
            "dependencies": sorted(set(section_deps)),
            "dependents": [],
        }

    # Fill dependents from dependencies
    for src, deps in {k: v["dependencies"] for k, v in sections.items()}.items():
        for d in deps:
            if d in sections:
                sections[d].setdefault("dependents", []).append(src)

    return {
        "modules": sections,
        "type": "latex",
    }


def scan_document(file_path: str) -> dict:
    """Auto-detect format and scan document structure."""
    ext = Path(file_path).suffix.lower()
    if ext in (".md", ".markdown"):
        return _scan_markdown(file_path)
    elif ext in (".tex", ".latex"):
        return _scan_latex(file_path)
    else:
        return {"error": f"Unsupported document format: {ext}"}
