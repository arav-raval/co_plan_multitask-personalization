"""
Usage:
    PYTHONPATH=src python scripts/visualize_recipe_dag.py \
        --recipe SweetCurry --human-type SpiceSpecific --out figures/dag.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize recipe DAG with preference coloring")
    p.add_argument("--recipe", default="SweetCurry",
                   help="Recipe name (e.g. SimpleDal, SweetCurry, UltraComplexFeast)")
    p.add_argument("--human-type", default="SpiceSpecific", dest="human_type",
                   help="Hidden HBM config name for true preferences")
    p.add_argument("--out", default=None,
                   help="Output path (e.g. figures/dag.pdf). If omitted, shows interactively.")
    p.add_argument("--dpi", type=int, default=300,
                   help="DPI for raster output (default: 300)")
    return p.parse_args()


def get_recipe(name: str):
    """Look up a recipe by name from the recipes module."""
    from multitask_personalization.envs.spices import recipes as R
    if hasattr(R, name):
        return getattr(R, name)
    upper = name.upper()
    if hasattr(R, upper):
        return getattr(R, upper)
    from multitask_personalization.envs.spices.spices_env import RecipeSpec
    for attr_name in dir(R):
        obj = getattr(R, attr_name)
        if isinstance(obj, RecipeSpec) and obj.name == name:
            return obj
    raise ValueError(f"Recipe '{name}' not found. Available: "
                     f"{[a for a in dir(R) if not a.startswith('_')]}")


def main() -> None:
    args = parse_args()

    from multitask_personalization.envs.spices.config.hidden_hbm_configs import (
        get_hidden_hbm_config,
    )

    recipe = get_recipe(args.recipe)
    hidden_cfg = get_hidden_hbm_config(args.human_type)
    theta = hidden_cfg.generate_theta(list(recipe.spices))

    G = nx.DiGraph()
    for spice in recipe.spices:
        G.add_node(spice)
    for spice, preds in recipe.predecessors.items():
        for p in preds:
            G.add_edge(p, spice)

    layers = recipe.layers()
    layer_index = {s: i for i, layer in enumerate(layers) for s in layer}
    nx.set_node_attributes(G, layer_index, "layer")

    HUMAN_COLOR = "#D5E8D4"
    ROBOT_COLOR = "#F8CECC"

    node_colors = [HUMAN_COLOR if theta[n] > 0 else ROBOT_COLOR for n in G.nodes]
    label_map = {node: node.replace("_", " ").title() for node in G.nodes}

    n_nodes = len(G.nodes)
    max_layer_size = max(len(layer) for layer in layers)
    n_layers = len(layers)

    x_spacing = 2.4
    y_spacing = 0.85

    pos = {}
    for layer_idx, layer in enumerate(layers):
        x = (layer_idx + 1) * x_spacing  
        total_height = (len(layer) - 1) * y_spacing
        y_start = total_height / 2.0
        for i, node in enumerate(sorted(layer)):
            pos[node] = (x, y_start - i * y_spacing)

    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]

    start_x = 0.0
    start_y = np.mean([pos[n][1] for n in roots])
    end_x = (n_layers + 1) * x_spacing
    end_y = np.mean([pos[n][1] for n in sinks])

    G.add_node("__start__")
    G.add_node("__end__")
    pos["__start__"] = (start_x, start_y)
    pos["__end__"] = (end_x, end_y)
    for r in roots:
        G.add_edge("__start__", r)
    for s in sinks:
        G.add_edge(s, "__end__")

    label_map["__start__"] = "Start"
    label_map["__end__"] = "End"

    node_size = 7500
    font_size = 12
    sentinel_size = 1200
    fig_w = max(10, (n_layers + 2) * 2.8)
    fig_h = max(4, max_layer_size * 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrowsize=14,
        arrowstyle="-|>",
        width=1.0,
        edge_color="#555555",
        connectionstyle="arc3,rad=0.06",
        min_source_margin=20,
        min_target_margin=20,
    )

    recipe_nodes = [n for n in G.nodes if n not in ("__start__", "__end__")]
    recipe_colors = [HUMAN_COLOR if theta[n] > 0 else ROBOT_COLOR for n in recipe_nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=recipe_nodes, ax=ax,
        node_size=node_size,
        node_color=recipe_colors,
        edgecolors="#333333",
        linewidths=1.0,
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=["__start__", "__end__"], ax=ax,
        node_size=sentinel_size,
        node_color="white",
        edgecolors="#333333",
        linewidths=1.5,
    )

    recipe_labels = {n: label_map[n] for n in recipe_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=recipe_labels, ax=ax,
        font_size=font_size,
        font_weight="bold",
        font_color="black",
        font_family="serif",
    )

    for node in ("__start__", "__end__"):
        x, y = pos[node]
        ax.text(x, y, label_map[node], ha="center", va="center",
                fontsize=font_size - 1, fontstyle="italic",
                fontfamily="serif", color="#333333")

    human_patch = mpatches.Patch(
        facecolor=HUMAN_COLOR, edgecolor="#333333", linewidth=1.0,
        label="Human-preferred ($\\theta_s > 0$)",
    )
    robot_patch = mpatches.Patch(
        facecolor=ROBOT_COLOR, edgecolor="#333333", linewidth=1.0,
        label="Robot-preferred ($\\theta_s \\leq 0$)",
    )

    profile_patch = mpatches.Patch(
        facecolor="white", edgecolor="white",
        label=f"Profile: {args.human_type}",
    )
    ax.legend(
        handles=[human_patch, robot_patch, profile_patch],
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=20,
        framealpha=0.95,
    )

    ax.set_title(
        f"Recipe: {recipe.name}",
        fontsize=20, fontweight="bold", fontfamily="serif", pad=14,
    )
    ax.axis("off")
    fig.tight_layout(pad=1.2)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved to {args.out}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
