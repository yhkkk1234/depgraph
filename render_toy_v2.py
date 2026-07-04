"""
Render a CLEAN, READABLE dependency diagram optimized for vision model OCR.
Large fonts, high contrast, no overlapping labels.
"""
import base64, io, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from scan_deps import scan_project

PROJECT = os.path.join(os.path.dirname(__file__), "test_project")
OUTPUT = os.path.join(os.path.dirname(__file__), "results", "toy_diagram_v2.png")

graph = scan_project(PROJECT)
modules = graph["modules"]

# Build graph
G = nx.DiGraph()
highlight_module = "models.task"
highlight_dependents = set(modules[highlight_module].get("dependents", []))

for mod_name in sorted(modules.keys()):
    label = mod_name.replace(".", "/") + ".py"
    G.add_node(mod_name, label=label)

for mod_name, info in modules.items():
    for dep in info.get("dependencies", []):
        if dep in modules:
            G.add_edge(mod_name, dep)

# Large, clean figure
fig, ax = plt.subplots(figsize=(20, 14), dpi=200)
pos = nx.spring_layout(G, k=4, iterations=100, seed=42)

node_colors = []
for node in G.nodes():
    if node == highlight_module:
        node_colors.append("#FF2222")  # Bright red
    elif node in highlight_dependents:
        node_colors.append("#FFD700")  # Gold
    else:
        node_colors.append("#E8E8E8")  # Light gray

# Draw with LARGE fonts
labels = {n: G.nodes[n]["label"] for n in G.nodes()}
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3500,
                       edgecolors="#333333", linewidths=2, ax=ax)
nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight="bold", ax=ax)
nx.draw_networkx_edges(G, pos, edge_color="#666666", arrows=True,
                       arrowsize=20, width=2, connectionstyle="arc3,rad=0.1", ax=ax)

# Legend
legend_items = [
    ("#FF2222", f"Changed: {highlight_module}"),
    ("#FFD700", "Directly affected"),
    ("#E8E8E8", "Other modules"),
]
patches = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#333", linewidth=1) for c, _ in legend_items]
ax.legend(patches, [l for _, l in legend_items], loc="lower right", fontsize=12, framealpha=0.9)

ax.set_title("Project Module Dependencies", fontsize=20, fontweight="bold")
ax.axis("off")
plt.tight_layout(pad=1)

# Save ultra-high-res
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, format="png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

print(f"Diagram saved: {OUTPUT}")
print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
print(f"Highlight: {highlight_module}")
print(f"Dependents: {highlight_dependents}")
