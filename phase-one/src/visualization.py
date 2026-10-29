"""Plotting, coupling-graph visualisers and equation renders."""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

COLORS = ["#4C72B0", "#DD8452"]


def draw_coupling_graph(proc, ax=None, title=None):
    """Draw the coupling graph; qubits used by the 4-qubit register are highlighted."""
    ax = ax or plt.gca()
    g = proc.graph
    pos = nx.kamada_kawai_layout(g) if proc.name.endswith("B") else {i: (i, 0) for i in g.nodes}
    used = set(proc.layout)
    colors = ["#DD8452" if n in used else "#CCCCCC" for n in g.nodes]
    nx.draw_networkx(g, pos, ax=ax, node_color=colors, node_size=650, font_size=10, edge_color="#555555")
    labels = {p: f"L{l}" for l, p in enumerate(proc.layout)}
    nx.draw_networkx_labels(g, {k: (v[0], v[1] - 0.18) for k, v in pos.items()}, labels=labels,
                            font_size=7, font_color="#8B0000", ax=ax)
    if not proc.name.endswith("B"):
        ax.set_ylim(-0.6, 0.6)          # leave room for the logical-qubit labels
    ax.set_title(title or f"{proc.name}: {proc.topology}")
    ax.axis("off")


def plot_dataset(X, y, path=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30)
    ax.set_title("make_moons (n=100, noise=0.1)"); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=200)
    return fig


def plot_kernel_heatmap(K, title, path=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=200)
    return fig


def _bar(ax, names, vals, title, ylabel):
    ax.bar(names, vals, color=COLORS)
    ax.set_title(title); ax.set_ylabel(ylabel)
    for k, v in enumerate(vals):
        ax.text(k, v, f"{v:.3g}", ha="center", va="bottom")


def plot_cost_comparison(df, path=None):
    """Depth, CX, time, estimated total time and kernel error for A vs B."""
    names = df["Processor"]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7)); ax = ax.ravel()
    _bar(ax[0], names, df["Mean depth"], "Compiled circuit depth (mean)", "Depth")
    _bar(ax[1], names, df["Mean CX gates"], "Compiled CX count (mean)", "CX gates")
    _bar(ax[2], names, df["Mean time / evaluation (s)"], "Time per kernel evaluation", "Seconds")
    _bar(ax[3], names, df["Est. total time (s)"], "Estimated time, full kernel", "Seconds")
    _bar(ax[4], names, df["Mean |noisy - ideal|"], "Noisy fidelity error vs ideal", "Mean abs error")
    ax[5].axis("off")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=200)
    return fig


def plot_routing_overhead(comp_df, path=None):
    """CX count and routing overhead per target for one demo circuit."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(comp_df["Target"], comp_df["CX"], color=["#999999"] + COLORS)
    ax[0].set_title("CX count (demo circuit)"); ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(comp_df["Target"], comp_df["Routing CX overhead"], color=["#999999"] + COLORS)
    ax[1].set_title("Routing CX overhead vs all-to-all"); ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=200)
    return fig


def render_equation(tex, path, fontsize=22):
    """Render a LaTeX math string to a tightly-cropped PNG using matplotlib mathtext."""
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${tex}$", fontsize=fontsize)
    fig.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.12, transparent=False, facecolor="white")
    plt.close(fig)
