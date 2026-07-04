"""
AST-based call graph analyzer.
Finds functions that are called from many different files — prime targets
for generating cross-module bug training data.
"""
import ast
import os
import re
from collections import defaultdict
from pathlib import Path


def _get_module(file_path: Path, project_root: Path) -> str:
    rel = file_path.relative_to(project_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace(".py", "")
    return ".".join(parts)


def _collect_direct_calls(node) -> list[str]:
    """Recursively collect function calls from node, skipping nested function bodies."""
    calls = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue  # Skip nested function bodies
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
        calls.extend(_collect_direct_calls(child))
    return calls


def _extract_defs_and_calls(file_path: Path) -> tuple[list[str], list[str]]:
    """Extract function definitions and function calls from a file."""
    defs = []
    calls = []
    ext = file_path.suffix.lower()

    # Python: AST
    if ext == ".py":
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return defs, calls
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    defs.append(node.name)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
        return defs, calls

    # JavaScript/TypeScript/JSX/TSX/Go/Rust: regex
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="gbk")
        except Exception:
            return defs, calls

    # Function definitions
    if ext in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"):
        for m in re.finditer(r"""(?:function|async\s+function)\s+(\w+)\s*\(""", text):
            name = m.group(1)
            if not name.startswith("_") and name not in ("if", "for", "while", "switch"):
                defs.append(name)
    elif ext == ".go":
        for m in re.finditer(r"""func\s+(\w+)\s*\(""", text):
            defs.append(m.group(1))
    elif ext == ".rs":
        for m in re.finditer(r"""fn\s+(\w+)\s*\(""", text):
            defs.append(m.group(1))
    else:
        # Generic: match any identifier followed by (
        for m in re.finditer(r"""\b(\w+)\s*\(""", text):
            name = m.group(1)
            if name[0].isupper() and not name.startswith("_") and name not in ("if", "for", "while", "switch", "return"):
                defs.append(name)

    # Function calls
    for m in re.finditer(r"""\b(\w+)\s*\(""", text):
        name = m.group(1)
        if name not in ("if", "for", "while", "switch", "return", "typeof", "console"):
            calls.append(name)

    return defs, calls


def analyze_call_graph(project_path: str) -> dict:
    """Build a call graph for a Python project.

    Returns:
        {
            "functions": {func_name: {"file": ..., "module": ..., "def_lines": ...}},
            "callers": {func_name: ["file1.py", "file2.py", ...]},
            "high_impact": [{"func": ..., "files": [...], "score": ...}, ...]
        }
    """
    root = Path(project_path).resolve()
    py_files = [f for f in root.rglob("*.*") if f.suffix in (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs")
                and "__pycache__" not in str(f) and "node_modules" not in str(f)]

    # Collect all function definitions
    functions = {}
    for fpath in py_files:
        mod = _get_module(fpath, root)
        defs, _ = _extract_defs_and_calls(fpath)
        for func_name in defs:
            if func_name not in functions:
                functions[func_name] = []
            functions[func_name].append({
                "file": str(fpath.relative_to(root)),
                "module": mod,
            })

    # Collect all call sites
    callers = defaultdict(set)
    for fpath in py_files:
        mod = _get_module(fpath, root)
        _, file_calls = _extract_defs_and_calls(fpath)
        for called_func in set(file_calls):
            if called_func in functions:
                callers[called_func].add(str(fpath.relative_to(root)))

    # Rank by impact (number of unique caller files)
    high_impact = []
    for func_name, caller_files in callers.items():
        n_callers = len(caller_files)
        # Filter: at least 3 caller files, exclude dunder methods
        if n_callers >= 3 and not func_name.startswith("__"):
            high_impact.append({
                "function": func_name,
                "defined_in": functions[func_name],
                "caller_files": sorted(caller_files),
                "caller_count": n_callers,
                "impact_score": n_callers,
            })

    high_impact.sort(key=lambda x: -x["impact_score"])

    return {
        "functions": {k: v for k, v in functions.items() if k in callers},
        "callers": {k: sorted(v) for k, v in callers.items()},
        "high_impact": high_impact,
        "total_functions": len(functions),
        "total_high_impact": len(high_impact),
    }


def suggest_bugs(call_graph: dict, source_cache: dict = None,
                 top_n: int = 10) -> list[dict]:
    """Generate bug scenarios from high-impact functions.

    Each suggestion includes the function source and affected files.
    """
    suggestions = []
    for info in call_graph["high_impact"][:top_n]:
        func_name = info["function"]
        # Get source of the function
        source = ""
        if source_cache:
            for def_info in info["defined_in"]:
                fpath = def_info["file"]
                if fpath in source_cache:
                    content = source_cache[fpath]
                    # Try to extract the function source (simple heuristic)
                    for line in content.split("\n"):
                        if f"def {func_name}(" in line:
                            source = line.strip()
                            break
                    if source:
                        break

        suggestions.append({
            "function": func_name,
            "defined_in": info["defined_in"],
            "caller_files": info["caller_files"],
            "caller_count": info["caller_count"],
            "signature": source,
            "bug_ideas": _generate_bug_ideas(func_name, source, info["caller_files"]),
        })

    return suggestions


def _generate_bug_ideas(func_name: str, signature: str, callers: list[str]) -> list[dict]:
    """Generate candidate bug scenarios for a function."""
    ideas = []

    # Count parameters
    params = signature.split("(", 1)[1].rstrip("):") if "(" in signature else ""
    param_list = [p.strip() for p in params.split(",") if p.strip() and "=" not in p]

    if param_list:
        # Bug type 1: make a parameter required
        ideas.append({
            "type": "param_required",
            "desc": f"Removed default value from parameter '{param_list[0]}' in {func_name}()",
            "change": f"Changed '{param_list[0]}' from optional to required",
        })

    if params and "=" in params:
        # Bug type 2: add a new required parameter
        ideas.append({
            "type": "new_required_param",
            "desc": f"Added new required parameter to {func_name}()",
            "change": f"Added required parameter 'strict: bool'",
        })

    # Bug type 3: change return type (affects all consumers)
    ideas.append({
        "type": "return_type_change",
        "desc": f"Changed return type of {func_name}() — now returns a dict instead of a tuple",
        "change": f"Return type changed from tuple to dict",
    })

    return ideas


if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else "."
    cg = analyze_call_graph(path)

    print(f"Total functions: {cg['total_functions']}")
    print(f"High-impact (3+ callers): {cg['total_high_impact']}")
    print()

    for info in cg["high_impact"][:15]:
        print(f"  {info['function']:25s} ← called from {info['caller_count']} files: "
              f"{', '.join(info['caller_files'][:5])}")
