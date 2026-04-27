from enum import Enum
from typing import Literal, Optional

import numpy as np
import torch
from torch_geometric.data import Data


class TargetMode(Enum):
    EDGES = "edges"
    NODES = "nodes"
    GLOBAL = "global"


class GraphDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset of particle trajectory graphs for training.

    Wraps a list of PyG Data objects and serves randomly sampled temporal
    subgraphs at each iteration. Each subgraph is drawn from a random
    time window of length Dt within a randomly chosen graph, with node
    coordinates centered per-subgraph and scaled by trajectory_span_std.

    Parameters
    ----------
    graph_dataset : list[Data]
        List of PyG Data objects, one per video or connected component.
    Dt : int
        Length of the temporal window in frames. Graphs with fewer than
        Dt nodes are discarded.
    dataset_size : int
        Number of subgraphs to serve per epoch, i.e. the value returned
        by __len__.
    trajectory_span_std : float, optional
        Standard deviation of coordinate spans across Dt-length trajectory
        windows (both axes combined), used to scale node coordinates after
        per-subgraph centering. Typically estimated via
        GraphFromTrajectories.from_tracks. Default is 1.0.
    transform : callable, optional
        A transform following the PyG transform interface
        (Data -> Data) applied to each subgraph before scaling.
        Default is None.
    target : {"edges", "nodes", "global"}, optional
        Determines what y contains in each returned subgraph.
        "edges" returns per-edge ground-truth labels, "global" returns
        the graph-level class label. Default is "edges".
    sample_balanced : bool, optional
        If True, graphs are sampled uniformly across class labels rather
        than uniformly across graphs. Useful for imbalanced datasets.
        Default is False.

    Methods
    -------
    center_and_scale_graph(graph)
        Center per-subgraph and scale by trajectory_span_std.
    __len__()
        Return the dataset size.
    __getitem__(idx)
        Sample and return a random temporal subgraph.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt: int,
        dataset_size: int,
        trajectory_span_std: Optional[float] = None,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "edges",
        sample_balanced: bool = False,
    ) -> None:
        self.Dt = Dt
        self.graph_dataset = [graph for graph in graph_dataset if len(graph.x) >= Dt]
        self.dataset_size = dataset_size
        self.trajectory_span_std = float(trajectory_span_std) if trajectory_span_std is not None else 1.0
        self.transform = transform
        self._target = None
        self.target = target
        self.labels = np.array([graph.graph_label.item() for graph in self.graph_dataset])
        self.sample_balanced = sample_balanced

    @property
    def target(self) -> TargetMode:
        return self._target

    @target.setter
    def target(self, value) -> None:
        self._target = TargetMode(value) if isinstance(value, str) else value

    def center_and_scale_graph(self, graph: Data) -> Data:
        """Center and scale node coordinates.

        Subtracts the per-subgraph mean (centering at origin) and divides
        by trajectory_span_std so that coordinates are on a consistent scale
        across recordings.

        Parameters
        ----------
        graph : Data
            PyG Data object whose x attribute contains node coordinates.

        Returns
        -------
        Data
            A cloned Data object with centered and scaled node coordinates.
        """
        graph = graph.clone()
        graph.x = (graph.x - graph.x.mean(dim=0)) / self.trajectory_span_std
        return graph

    def __len__(self) -> int:
        """Return the number of subgraphs served per epoch.

        Returns
        -------
        int
            The dataset size passed at construction.
        """
        return self.dataset_size

    def __getitem__(self, idx: int) -> Data:
        """Sample a random temporal subgraph from the dataset.

        Selects a random graph, then samples a random time window of
        length Dt. Nodes and edges outside the window are masked out
        and edge indices are reindexed to be contiguous. The subgraph
        is then transformed and scaled.

        Parameters
        ----------
        idx : int
            Unused; sampling is random and independent of the index.

        Returns
        -------
        Data
            A PyG Data object containing:
                - x : centered and scaled node coordinates, shape (N, 2)
                - edge_index : reindexed edge connectivity, shape (2, E)
                - edge_attr : edge feature vectors, shape (E, 3)
                - distance : normalised distances, shape (E, 1)
                - y : target labels, shape depends on target mode
        """
        if self.sample_balanced:
            graph_label = np.random.choice(np.unique(self.labels))
            selected_idx = np.random.choice(np.where(self.labels == graph_label)[0])
        else:
            selected_idx = np.random.randint(0, len(self.graph_dataset) - 1)

        graph = self.graph_dataset[selected_idx]
        frames, edge_index = graph.frames, graph.edge_index

        node_mask = torch.tensor([])
        while node_mask.sum() < 2:
            max_frame = frames.max()
            start_frame = np.random.choice(frames[frames <= (max_frame - self.Dt + 1)].numpy())
            end_frame = start_frame + self.Dt
            node_mask = (frames >= start_frame) & (frames < end_frame)

        frame_pairs = torch.stack(
            [frames[edge_index[0, :]], frames[edge_index[1, :]]],
            dim=-1,
        )
        edge_mask = (frame_pairs >= start_frame) & (frame_pairs < end_frame)
        edge_mask = edge_mask.all(dim=-1)

        edge_index = edge_index[:, edge_mask] - edge_index[:, edge_mask].min()

        match self.target:
            case TargetMode.EDGES:
                y = graph.y[edge_mask]
            case TargetMode.GLOBAL:
                y = graph.graph_label
            case TargetMode.NODES:
                raise NotImplementedError("Node-level targets are not yet supported.")

        return_graph = Data(
            x=graph.x[node_mask],
            edge_index=edge_index,
            edge_attr=graph.edge_attr[edge_mask],
            distance=graph.edge_attr[edge_mask, 0:1],
            y=y,
        )

        if self.transform:
            return_graph = self.transform(return_graph)
        return_graph = self.center_and_scale_graph(return_graph)

        return return_graph
