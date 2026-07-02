"""
Render project dependency graph as PNG using networkx + matplotlib.
Pure local rendering, no external API dependency.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def render_dependency_graph(graph: dict, highlight_module: str = None,
                            output_path: str = None,
                            as_data_url: bool = False) -> str:
    """Render dependency graph to PNG.

    Args:
        graph: dependency graph dict from scan_deps
        highlight_module: module to highlight as changed
        output_path: if set, save PNG to this path
        as_data_url: if True, return 'data:image/png;base64,...' format.
                     if False, return raw base64 string (for MiMo/OpenAI compat).

    Returns:
        Raw base64 string or data URL, or file path if output_path set.
    """
    G = nx.DiGraph()
    modules = graph.get("modules", {})

    highlight_dependents = set()
    if highlight_module and highlight_module in modules:
        highlight_dependents = set(modules[highlight_module].get("dependents", []))

    for mod_name in sorted(modules.keys()):
        label = mod_name.replace(".", "\n.")
        G.add_node(mod_name, label=label)

    for mod_name, info in modules.items():
        for dep in info.get("dependencies", []):
            if dep in modules:
                G.add_edge(mod_name, dep)

    plt.figure(figsize=(14, 10), dpi=120)
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    node_colors = []
    for node in G.nodes():
        if node == highlight_module:
            node_colors.append("#ff4444")
        elif node in highlight_dependents:
            node_colors.append("#ffd43b")
        else:
            node_colors.append("#e9ecef")

    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800,
                           edgecolors="#495057", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="#adb5bd", arrows=True,
                           arrowsize=15, width=1.2, connectionstyle="arc3,rad=0.1")

    if highlight_module:
        legend_items = [
            ("#ff4444", f"Changed: {highlight_module}"),
            ("#ffd43b", "Directly affected"),
            ("#e9ecef", "Other modules"),
        ]
        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#495057", linewidth=1)
                          for c, _ in legend_items]
        plt.legend(legend_patches, [l for _, l in legend_items],
                   loc="lower right", fontsize=8, framealpha=0.9)

    plt.title("Project Module Dependency Graph", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        buf.seek(0)
        with open(output_path, "wb") as f:
            f.write(buf.read())
        return output_path

    buf.seek(0)
    raw_b64 = base64.b64encode(buf.read()).decode()
    if as_data_url:
        return f"data:image/png;base64,{raw_b64}"
    return raw_b64


def mermaid_to_data_url(mermaid_code: str) -> str:
    """Compatibility shim: this module no longer uses Mermaid.
    Call render_dependency_graph with a graph dict instead."""
    raise NotImplementedError("Use render_dependency_graph() with a graph dict")


if __name__ == "__main__":
    import json, sys
    graph_json = sys.stdin.read()
    graph = json.loads(graph_json)
    out = sys.argv[1] if len(sys.argv) > 1 else "dependency_graph.png"
    path = render_dependency_graph(graph, highlight_module="models.task", output_path=out)
    print(f"Saved to: {path}")
