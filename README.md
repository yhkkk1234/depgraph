# depgraph

**Visual dependency graphs for AI-assisted code analysis.**

Generate color-coded module dependency diagrams + text keys + prompt snippets, then inject into any LLM conversation to improve cross-module bug-fixing accuracy.

[![Paper](https://img.shields.io/badge/paper-preprint-blue)](https://github.com/yhkkk1234/depgraph/blob/main/preprint.tex)

## Quick Start

```bash
# Install
pip install matplotlib networkx

# Generate graph for any project
python depgraph.py /path/to/project -o ./output/

# Inject directly into prompt
python depgraph.py /path/to/project -i
```

Outputs:
- `dependency_graph.png` — Red = changed module, yellow = dependents
- `dependency_key.txt` — 3-line text legend (module → dependents)
- `prompt_snippet.txt` — Ready to paste into AI conversation

## Supported Languages

Python, JavaScript, TypeScript, JSX, TSX, Go, Rust.

## How It Works

1. AST/regex-based import scanning across all source files
2. Builds directed dependency graph `G = (V, E)`
3. Renders color-coded network diagram (networkx + matplotlib)
4. Generates compact text key for models with limited OCR
5. Outputs prompt-ready snippet: diagram + key + bug description

## Experiments

This tool was developed as part of research into improving LLM "global perspective" when fixing cross-module bugs. Key findings:

| Input Mode | File Coverage |
|-----------|--------------|
| Text only | 0% |
| **Diagram + 3-line Key** | **100%** |

Full paper: [`preprint.tex`](preprint.tex). Compiles on Overleaf.

## Citation

```bibtex
@misc{depgraph2026,
  title={Visual Dependency Graphs Improve LLM Cross-Module Bug Fixing},
  year={2026},
  url={https://github.com/yhkkk1234/depgraph}
}
```
