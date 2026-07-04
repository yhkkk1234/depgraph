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
import json, os, sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from scan_deps import scan_project
from render_diagram import render_dependency_graph


def generate_graph(project: str, module: str = None) -> str:
    """Generate dependency graph + text key for a project/module."""
    if not os.path.isdir(project):
        return f"Error: project path '{project}' not found."

    graph = scan_project(project)
    modules = graph.get("modules", {})

    if not module:
        best = max(modules.items(), key=lambda x: len(x[1].get("dependents", [])))
        module = best[0]

    if module not in modules:
        similar = [m for m in modules if module.replace(".py", "").replace("/", ".") in m.replace(".py", "").replace("/", ".")]
        if similar:
            module = similar[0]
        else:
            return f"Error: module '{module}' not found. Available: {', '.join(sorted(modules.keys())[:10])}..."

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
    lines.append(f"Summary: changing {module} affects {len(rdeps)} dependents.")

    return "\n".join(lines)


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
                "description": "Generate a module dependency graph for a project. Returns which files depend on a given module, showing the full impact surface of any code change. Use this when debugging cross-module bugs to identify ALL affected files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "Path to the project directory"},
                        "module": {"type": "string", "description": "The module to analyze (e.g. 'models.task' or 'utils')"},
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
        module = args.get("module")
        result_text = generate_graph(project, module)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result_text}]},
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
