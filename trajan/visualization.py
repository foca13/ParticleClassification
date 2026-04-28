import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path


def plot_confusion_matrix(cm_df: pd.DataFrame, display_labels: list) -> plt.Figure:
    """Plot a confusion matrix and return the figure."""
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=cm_df.values, display_labels=display_labels
    ).plot(ax=ax)
    return fig


def plot_classification_report(report_df: pd.DataFrame) -> plt.Figure:
    """Plot a classification report table and return the figure."""
    accuracy = report_df.loc["accuracy", "f1-score"]
    plot_df = report_df.drop("accuracy")

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=plot_df.values,
        rowLabels=plot_df.index,
        colLabels=plot_df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.text(
        -0.164, 0.1,
        f"accuracy: {accuracy:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    return fig


def save_classification_report(report_df: pd.DataFrame, path: Path) -> None:
    """Save a classification report to CSV, appending the accuracy row cleanly."""
    accuracy = report_df.loc["accuracy", "f1-score"]
    report_df.drop("accuracy").to_csv(path)
    with open(path, "a") as f:
        f.write(f"\naccuracy,{accuracy:.2f}")


def plot_graph(graph: Data, to_undirected: bool = True, axis: bool = True, node_size: int = 30):
    """Plot a PyG Data object as a networkx graph.

    Parameters
    ----------
    graph : Data
        PyG Data object with edge_index and optionally x containing
        node coordinates in the first two columns.
    to_undirected : bool, optional
        If True, edges are treated as undirected. Default is True.
    node_size : int, optional
        Size of the nodes in the plot. Default is 30.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes containing the plot.
    """

    fig, ax = plt.subplots()

    G = to_networkx(graph, to_undirected=to_undirected)

    # Use node positions from x if available, otherwise spring layout
    if graph.x is not None:
        pos = {i: graph.x[i, :2].numpy() for i in range(graph.num_nodes)}
    else:
        pos = nx.spring_layout(G)

    nx.draw_networkx(G, pos=pos, ax=ax, node_size=node_size, node_color="steelblue", with_labels=False)

    if axis:
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    return fig, ax