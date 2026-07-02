"""
Generate text legend from dependency graph JSON.
Provides accurate module names as text, to accompany the visual diagram.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from scan_deps import scan_project


def generate_legend(graph: dict, highlight: str = None) -> str:
    """Generate a human-readable text legend for a dependency graph."""
    modules = graph.get("modules", {})
    if not highlight or highlight not in modules:
        return ""

    changed = modules[highlight]
    dependents = changed.get("dependents", [])
    all_deps = changed.get("dependencies", [])

    lines = []
    lines.append(f"## Module Dependency Legend\n")
    lines.append(f"**{highlight}** (CHANGED)")

    if dependents:
        lines.append(f"  Used by (direct dependents):")
        for d in dependents:
            lines.append(f"    - {d}")

    if all_deps:
        lines.append(f"  Depends on:")
        for d in all_deps:
            lines.append(f"    - {d}")

    return "\n".join(lines)


def generate_full_legend(graph: dict, highlight: str = None) -> str:
    """Generate complete legend with ALL module relationships."""
    modules = graph.get("modules", {})
    lines = ["## Full Module Dependencies\n"]
    
    for mod_name in sorted(modules.keys()):
        info = modules[mod_name]
        deps = info.get("dependencies", [])
        rdeps = info.get("dependents", [])
        marker = " **[CHANGED]**" if mod_name == highlight else ""
        lines.append(f"\n**{mod_name}**{marker}")
        if deps:
            lines.append(f"  depends on: {', '.join(deps)}")
        if rdeps:
            lines.append(f"  used by: {', '.join(rdeps)}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    proj = sys.argv[1] if len(sys.argv) > 1 else "."
    highlight = sys.argv[2] if len(sys.argv) > 2 else None

    g = scan_project(proj)
    if highlight:
        print(generate_legend(g, highlight))
    print()
    print(generate_full_legend(g, highlight))
