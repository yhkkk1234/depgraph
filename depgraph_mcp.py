#!/usr/bin/env python
"""
depgraph MCP Server — expose dependency graph generation as an MCP tool.
Register in opencode.json MCP config to enable spontaneous tool calling.

Usage in opencode.json:
  "mcp": {
    "depgraph": {
      "type": "local",
      "command": ["python", "path/to/depgraph_mcp.py", "--project-dir", "/your/project"],
      "enabled": true
    }
  }
"""
import json, os, sys, ast, re
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from scan_deps import scan_project
from render_diagram import render_dependency_graph
from call_graph import analyze_call_graph


def generate_graph(project: str, module: str = None, function: str = None, intra: str = None) -> tuple[str, str]:
    """Generate text key + base64 PNG diagram.
    
    Args:
        project: Project directory path
        module: Module to analyze (e.g. 'models.task')
        function: Optional, function name for call-site analysis
        intra: Optional, single file path for intra-file call analysis
    Returns: (text_report, image_b64_or_none)
    """

    # ── Intra-file analysis ──
    if intra:
        t, _ = _intra_file_analysis(intra, function)
        return t, None

    if not os.path.isdir(project):
        return f"Error: project path '{project}' not found.", None

    # ── Function-level call analysis ──
    if function:
        t, _ = _function_analysis(project, function)
        return t, None

    # ── Module-level dependency analysis ──
    graph = scan_project(project)
    modules = graph.get("modules", {})

    if not module:
        best = max(modules.items(), key=lambda x: len(x[1].get("dependents", [])))
        module = best[0]

    if module not in modules:
        similar = [m for m in modules if module in m]
        if similar:
            module = similar[0]
        else:
            return f"Error: module '{module}' not found. Available: {', '.join(sorted(modules.keys())[:10])}...", None

    info = modules[module]
    deps = info.get("dependencies", [])
    rdeps = info.get("dependents", [])

    lines = []
    lines.append(f"=== Dependency Graph for {module} ===")
    lines.append(f"")
    if deps:
        lines.append(f"Depends on ({len(deps)}):")
        for d in deps:
            lines.append(f"  - {d}")
    if rdeps:
        lines.append(f"Used by ({len(rdeps)}):")
        for d in rdeps:
            lines.append(f"  - {d}")
    if not rdeps and not deps:
        lines.append("No dependencies found.")
    lines.append(f"")
    lines.append(f"Summary: changing {module} affects {len(rdeps)} dependent modules.")
    text = "\n".join(lines)

    img_b64 = None
    try:
        img_b64 = render_dependency_graph(graph, highlight_module=module)
    except Exception:
        pass

    return text, img_b64


def _function_analysis(project: str, function: str) -> tuple[str, str]:
    """Analyze which files call a specific function."""
    cg = analyze_call_graph(project)
    lines = []
    lines.append(f"=== Function Call Analysis: {function}() ===\n")

    # Find function definition
    for hi in cg.get("high_impact", []):
        if hi["function"] == function:
            callers = hi["caller_files"]
            lines.append(f"Defined in: {', '.join(d['file'] for d in hi.get('defined_in', []))}")
            lines.append(f"Called from ({len(callers)} files):")
            for f in callers:
                lines.append(f"  - {f}")
            lines.append(f"\nTotal: {len(callers)} caller files.")
            return "\n".join(lines), None

    # Try to find in all functions
    for func_name, callers in cg.get("callers", {}).items():
        if func_name == function:
            lines.append(f"Called from ({len(callers)} files):")
            for f in sorted(callers):
                lines.append(f"  - {f}")
            return "\n".join(lines), None

    lines.append(f"Function '{function}' not found in call graph.")
    lines.append("Try checking the module-level dependencies instead.")
    return "\n".join(lines), None


def _intra_file_analysis(file_path: str, function: str = None) -> str:
    """Analyze function calls within a single file. Supports Python, JS/TS, C#, and others."""
    try:
        text = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        try:
            text = Path(file_path).read_text(encoding="gbk")
        except Exception:
            return f"Error: cannot read {file_path}"

    ext = Path(file_path).suffix.lower()

    # Python: use AST
    if ext in (".py",):
        return _intra_python(text, function)
    # JS/TS/C#/Java: use regex
    else:
        return _intra_regex(text, function, ext)


def _intra_python(text: str, function: str = None) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "Error: cannot parse Python file."

    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            call_list = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                    call_list.append(child.func.id)
            # Remove calls from nested function bodies (keep only direct calls)
            direct_set = set(call_list)
            for child in ast.walk(node):
                if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for grandchild in ast.walk(child):
                        if isinstance(grandchild, ast.Call) and isinstance(grandchild.func, ast.Name):
                            direct_set.discard(grandchild.func.id)
            funcs[node.name] = {"lineno": node.lineno, "calls": sorted(direct_set)}

    return _format_intra_result(funcs, function, "Python")


def _intra_regex(text: str, function: str = None, ext: str = "") -> str:
    """Generic function detection via regex for non-Python files."""
    # Match function/method definitions
    patterns = [
        r"(?:function|def|void|async|public|private|static|protected)\s+(\w+)\s*\(",
        r"(\w+)\s*=\s*(?:function|async)\s*\(",
        r"(\w+)\s*:\s*function\s*\(",
        r"(\w+)\s*=\s*\([^)]*\)\s*=>",
        r"const\s+(\w+)\s*=",
    ]
    funcs = {}
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            name = m.group(1)
            if name not in ("if", "for", "while", "switch", "catch", "return"):
                if name not in funcs:
                    funcs[name] = {"lineno": text[:m.start()].count("\n") + 1, "start": m.start(), "calls": []}

    # For each function, find its body using brace matching, then extract direct calls
    for name, info in funcs.items():
        # Find opening brace after function definition
        brace_start = text.find("{", info["start"])
        if brace_start < 0:
            continue
        # Find matching closing brace
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end < 0:
            brace_end = info["start"] + 3000  # fallback

        body = text[brace_start:brace_end]
        # Remove content of nested function definitions before counting calls
        body_clean = _remove_nested_funcs(body)
        # Extract calls from cleaned body
        calls = []
        for cm in re.finditer(r"\b(\w+)\s*\(", body_clean):
            cn = cm.group(1)
            if cn not in ("if", "for", "while", "switch", "catch", "return", "typeof", "console",
                          "new", "throw", "void", "delete", "import", "export", "require",
                          "setTimeout", "setInterval", "clearTimeout", "clearInterval") \
               and not cn[0].isupper():  # Skip class/constructor calls
                if cn not in calls and cn != name:
                    calls.append(cn)
        funcs[name]["calls"] = calls

    lang = ext.upper() if ext else "Unknown"
    return _format_intra_result(funcs, function, lang)


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Find the position of the closing brace matching the opening brace at open_pos."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _remove_nested_funcs(body: str) -> str:
    """Remove content inside nested function bodies from the text."""
    # Match nested function/arrow definitions and remove their bodies
    result = body
    nested_pats = [
        r"(?:function|async\s+function)\s*\w*\s*\(.*?\)\s*\{",
        r"\([^)]*\)\s*=>\s*\{",
        r"\w+\s*=\s*\([^)]*\)\s*=>\s*\{",
    ]
    for pat in nested_pats:
        for m in re.finditer(pat, result):
            brace = result.find("{", m.start())
            if brace >= 0:
                end = _find_matching_brace(result, brace)
                if end >= 0:
                    # Replace nested body with whitespace (preserve position offsets)
                    result = result[:brace+1] + " " * (end - brace - 1) + result[end:]
    return result


def _format_intra_result(funcs: dict, function: str = None, lang: str = "") -> str:
    lines = []
    lines.append(f"=== Intra-File Analysis ({lang}) ===\n")
    lines.append(f"Functions defined: {len(funcs)}")
    sorted_funcs = sorted(funcs.items(), key=lambda x: x[1]["lineno"])
    for name, info in sorted_funcs:
        lines.append(f"  line {info['lineno']}: {name}()  calls: {info['calls'][:5]}")
        if len(info["calls"]) > 5:
            lines.append(f"    ... +{len(info['calls'])-5} more")

    if function:
        lines.append(f"\nCallers of {function}():")
        callers = [name for name, info in funcs.items() if function in info.get("calls", [])]
        if callers:
            for c in sorted(callers, key=lambda x: funcs[x]["lineno"]):
                lines.append(f"  line {funcs[c]['lineno']}: {c}()")
        else:
            lines.append("  No direct callers found in this file.")
    else:
        lines.append(f"\nTop 10 most-called functions:")
        call_counts = {}
        for fn, info in funcs.items():
            for c in info["calls"]:
                call_counts[c] = call_counts.get(c, 0) + 1
        top = sorted(call_counts.items(), key=lambda x: -x[1])[:10]
        for fn, count in top:
            lines.append(f"  {fn}(): called by {count} function(s)")

    return "\n".join(lines), None


# ── MCP stdio protocol (minimal JSON-RPC) ──

def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "depgraph", "version": "1.0"},
                "capabilities": {"tools": {}},
            }
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": [{
                "name": "depgraph",
                "description": "REQUIRED first step for any cross-module bug fix: scan the project to identify which files depend on the changed module. Returns a complete dependency list showing the full impact surface. MUST be called before proposing any code changes — never skip this tool when debugging interface changes, signature modifications, or module refactoring.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "Path to the project directory"},
                        "module": {"type": "string", "description": "The module to analyze (e.g. 'models.task' or 'utils'). Use this for cross-module impact analysis."},
                        "function": {"type": "string", "description": "Optional: a specific function name. If provided, finds ALL files that call this function across the project."},
                        "intra": {"type": "string", "description": "Optional: path to a single file. If provided, analyzes internal function calls within that file only."},
                    },
                    "required": ["project"],
                },
            }]},
        }

    if method == "tools/call":
        params = req.get("params", {})
        args = params.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        project = args.get("project", ".")
        # Coerce params: MCP clients may pass arrays or nested objects
        def _as_str(v):
            if v is None: return None
            if isinstance(v, str): return v
            if isinstance(v, (list, tuple)): return str(v[0]) if v else None
            return str(v)
        module = _as_str(args.get("module"))
        function = _as_str(args.get("function"))
        intra = _as_str(args.get("intra"))
        result_text, img_b64 = generate_graph(project, module, function, intra)
        content = [{"type": "text", "text": result_text}]
        if img_b64:
            content.append({"type": "image", "data": img_b64, "mimeType": "image/png"})
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": content},
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


def main():
    # Parse --project-dir from command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=".")
    args = parser.parse_args()

    default_project = os.path.abspath(args.project_dir)

    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            resp = handle_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": req.get("id") if 'req' in dir() else None,
                       "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
