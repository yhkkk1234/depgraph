#!/usr/bin/env python
"""
depgraph — Generate visual dependency graph + text key for AI code analysis.

Usage:
  depgraph <project_path> [--highlight MODULE] [--output DIR]
  depgraph --inject <project_path>  # output prompt-ready snippet

Outputs:
  - dependency_graph.png (visual diagram, red=changed, yellow=dependents)
  - dependency_key.txt  (text legend: module -> dependents mapping)
  - prompt_snippet.txt  (ready to paste into AI conversation)
"""
import argparse, json, os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scan_deps import scan_project, graph_to_mermaid
from render_diagram import render_dependency_graph
from legend_gen import generate_full_legend
from doc_scanner import scan_document


def main():
    parser = argparse.ArgumentParser(description="Generate dependency graph + text key for AI code analysis")
    parser.add_argument("project", help="Path to Python project directory or document file")
    parser.add_argument("--highlight", "-H", help="Module/section to highlight as changed")
    parser.add_argument("--output", "-o", default=".", help="Output directory (default: current dir)")
    parser.add_argument("--inject", "-i", action="store_true", help="Output prompt-ready snippet to stdout")
    parser.add_argument("--doc", "-d", action="store_true", help="Scan a document (Markdown/LaTeX) instead of code")
    args = parser.parse_args()

    target = Path(args.project).resolve()

    if args.doc:
        # Document mode
        if not target.is_file():
            print(f"Error: --doc requires a file path, got a directory", file=sys.stderr)
            sys.exit(1)
        print(f"Scanning document {target}...", file=sys.stderr)
        graph = scan_document(str(target))
        if "error" in graph:
            print(f"Error: {graph['error']}", file=sys.stderr)
            sys.exit(1)
    else:
        # Code mode
        if not target.is_dir():
            print(f"Error: {args.project} is not a directory (use --doc for files)", file=sys.stderr)
            sys.exit(1)
        print(f"Scanning {target}...", file=sys.stderr)
        graph = scan_project(str(target))
    modules = graph.get("modules", {})
    print(f"  Found {len(modules)} modules", file=sys.stderr)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate diagram PNG
    png_path = out_dir / "dependency_graph.png"
    highlight = args.highlight

    # Auto-detect highlight if not specified
    if not highlight:
        # Find module with most dependents
        best = max(modules.items(), key=lambda x: len(x[1].get("dependents", [])))
        highlight = best[0]
        print(f"  Auto-highlight: {highlight} ({len(best[1].get('dependents', []))} dependents)", file=sys.stderr)

    render_dependency_graph(graph, highlight_module=highlight, output_path=str(png_path))
    print(f"  Diagram: {png_path}", file=sys.stderr)

    # 2. Generate text key
    key_path = out_dir / "dependency_key.txt"
    dependents = modules.get(highlight, {}).get("dependents", [])
    deps = modules.get(highlight, {}).get("dependencies", [])

    key_lines = []
    key_lines.append(f"## Dependency Key")
    key_lines.append(f"{highlight} (CHANGED)")
    if dependents:
        key_lines.append(f"  Used by: {', '.join(dependents)}")
    if deps:
        key_lines.append(f"  Depends on: {', '.join(deps)}")

    key_text = "\n".join(key_lines)
    key_path.write_text(key_text, encoding="utf-8")
    print(f"  Key: {key_path}", file=sys.stderr)

    # 3. Generate prompt snippet
    snippet_path = out_dir / "prompt_snippet.txt"
    snippet = f"""## Project Dependency Info

{key_text}

DIAGRAM: dependency_graph.png (red={highlight}, yellow=dependents)

Based on the diagram and key above, identify ALL files affected by a change to {highlight}."""
    snippet_path.write_text(snippet, encoding="utf-8")
    print(f"  Snippet: {snippet_path}", file=sys.stderr)

    if args.inject:
        print(snippet)

    # Summary
    print(f"\nDone. Ready to inject into AI conversation.", file=sys.stderr)
    print(f"  paste: {snippet_path}", file=sys.stderr)
    print(f"  image: {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
