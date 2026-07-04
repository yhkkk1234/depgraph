"""
Multi-language project dependency scanner.
Supports: Python (.py), JavaScript/TypeScript (.js/.ts/.jsx/.tsx), Go (.go), Rust (.rs)
"""
import ast, json, os, re
from pathlib import Path
from collections import defaultdict


def _safe_read(fp: Path) -> str:
    """Read file with encoding fallback (UTF-8 -> GBK -> Latin-1)."""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return fp.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


# ── Module name resolution per language ──

def _module_py(fp: Path, root: Path) -> str:
    rel = fp.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)

def _module_path(fp: Path, root: Path) -> str:
    rel = fp.relative_to(root)
    return str(rel.with_suffix("")).replace("\\", "/")


# ── Import extraction per language ──

def _imports_py(fp: Path) -> list[str]:
    imports = []
    try:
        tree = ast.parse(_safe_read(fp))
    except (SyntaxError, UnicodeDecodeError):
        return imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def _imports_js(fp: Path, root: Path = None) -> list[str]:
    """Extract local imports from JS/TS files."""
    imports = []
    try:
        text = _safe_read(fp)
    except UnicodeDecodeError:
        return imports
    # import X from './path' or require('./path')
    for m in re.finditer(r"""(?:from\s+['"]|require\s*\(\s*['"])(\.\.?/[^'"]+)""", text):
        path = m.group(1)
        # Resolve relative to absolute
        resolved = (fp.parent / path).resolve()
        # Use project root if provided, otherwise fall back to parent^3
        base = root if root else fp.parent.parent.parent
        try:
            imports.append(_module_path(resolved, base))
        except (ValueError, OSError):
            pass
    return imports

def _imports_go(fp: Path) -> list[str]:
    """Extract internal imports from Go files."""
    imports = []
    try:
        text = _safe_read(fp)
    except UnicodeDecodeError:
        return imports
    module = ""
    m = re.search(r"^module\s+(\S+)", text, re.MULTILINE)
    if m:
        module = m.group(1)
    for m in re.finditer(r"""import\s+(?:[^(]|\("([^"]+)"\)|\s*"([^"]+)")""", text, re.DOTALL):
        pkg = m.group(1) or m.group(2) or ""
        if module and pkg.startswith(module):
            imports.append(pkg.replace(module + "/", ""))
    return imports

def _imports_rs(fp: Path) -> list[str]:
    """Extract internal module imports from Rust files."""
    imports = []
    try:
        text = _safe_read(fp)
    except UnicodeDecodeError:
        return imports
    for m in re.finditer(r"(?:mod|use)\s+(crate::)?([a-zA-Z_][\w:]*)", text):
        path = m.group(2).replace("::", "/")
        imports.append(path)
    return imports


# ── Language config ──

LANG_CONFIG = {
    ".py":  {"resolver": _module_py,   "extractor": _imports_py,  "globs": ["*.py"]},
    ".js":  {"resolver": _module_path, "extractor": _imports_js,  "globs": ["*.js", "*.mjs"]},
    ".ts":  {"resolver": _module_path, "extractor": _imports_js,  "globs": ["*.ts"]},
    ".jsx": {"resolver": _module_path, "extractor": _imports_js,  "globs": ["*.jsx"]},
    ".tsx": {"resolver": _module_path, "extractor": _imports_js,  "globs": ["*.tsx"]},
    ".go":  {"resolver": _module_path, "extractor": _imports_go,  "globs": ["*.go"]},
    ".rs":  {"resolver": _module_path, "extractor": _imports_rs,  "globs": ["*.rs"]},
}


def scan_project(project_root: str) -> dict:
    """Scan a project for cross-file dependencies. Auto-detects languages."""
    root = Path(project_root).resolve()
    modules = {}
    reverse_deps = defaultdict(set)

    # Collect files by language
    all_files = []
    for ext, cfg in LANG_CONFIG.items():
        for g in cfg["globs"]:
            all_files.extend(root.rglob(g))

    all_files = [f for f in all_files if "__pycache__" not in str(f) and "node_modules" not in str(f)]

    # Determine resolver/extractor per file
    def get_cfg(fp):
        return LANG_CONFIG.get(fp.suffix, LANG_CONFIG.get("".join(fp.suffixes[-2:]), None))

    # Map files to module names
    file_to_module = {}
    all_module_names = set()
    for fpath in all_files:
        cfg = get_cfg(fpath)
        if cfg:
            mod_name = cfg["resolver"](fpath, root)
            file_to_module[fpath] = mod_name
            all_module_names.add(mod_name)

    # Extract dependencies
    for fpath in all_files:
        cfg = get_cfg(fpath)
        if not cfg:
            continue
        mod_name = file_to_module[fpath]
        extractor = cfg["extractor"]
        # Pass root to extractors that need it
        try:
            file_imports = extractor(fpath, root) if extractor.__name__ == "_imports_js" else extractor(fpath)
        except TypeError:
            file_imports = extractor(fpath)

        internal_deps = []
        for imp in file_imports:
            best = None
            for other_name in all_module_names:
                imp_norm = imp.replace("\\", "/")
                other_norm = other_name.replace("\\", "/")
                # Exact match, or imp is a child path of other_name
                if (imp_norm == other_norm or 
                    imp_norm.startswith(other_norm + "/")):
                    if best is None or len(other_name) > len(best):
                        best = other_name
            if best and best != mod_name:
                internal_deps.append(best)
                reverse_deps[best].add(mod_name)

        modules[mod_name] = {
            "file": str(fpath.relative_to(root)),
            "dependencies": sorted(set(internal_deps)),
            "dependents": [],
        }

    for mod_name, deps_set in reverse_deps.items():
        if mod_name in modules:
            modules[mod_name]["dependents"] = sorted(deps_set)

    return {
        "project_root": str(root),
        "modules": modules,
    }


def graph_to_mermaid(graph: dict, highlight: str = None) -> str:
    """Convert dependency graph to Mermaid diagram code."""
    lines = ["graph TD"]
    modules = graph["modules"]
    node_ids = {}

    for i, mod_name in enumerate(sorted(modules.keys())):
        node_id = f"N{i}"
        node_ids[mod_name] = node_id
        display = mod_name.replace(".", "_")
        style = ""
        if highlight and mod_name == highlight:
            style = ":::highlight"
        elif highlight and highlight in modules and mod_name in modules[highlight].get("dependents", []):
            style = ":::affected"
        lines.append(f"    {node_id}[{display}]{style}")

    # Add edges
    for mod_name, info in sorted(modules.items()):
        src_id = node_ids[mod_name]
        for dep in info["dependencies"]:
            if dep in node_ids:
                dst_id = node_ids[dep]
                lines.append(f"    {src_id} --> {dst_id}")

    # Style definitions
    lines.append("")
    lines.append("    classDef highlight fill:#ff6b6b,stroke:#c92a2a,color:#fff")
    lines.append("    classDef affected fill:#ffd43b,stroke:#fab005,color:#333")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "."
    graph = scan_project(project)
    print("=== JSON ===")
    print(json.dumps(graph, indent=2, ensure_ascii=False))
    print("\n=== Mermaid ===")
    highlight = sys.argv[2] if len(sys.argv) > 2 else None
    print(graph_to_mermaid(graph, highlight))
