from enum import Enum
from typing import Literal, Optional, Tuple

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
    time window of length sampled uniformly from Dt_range within a randomly
    chosen graph. Node coordinates are centered per-subgraph and normalized
    by the mean step size of that subgraph, making the representation
    unit-agnostic while preserving scale information in graph-level
    attributes (local_scale, normalized_rg).

    Parameters
    ----------
    graph_dataset : list[Data]
        List of PyG Data objects, one per video or connected component.
        Graphs with fewer than Dt_range[0] nodes are discarded.
    Dt_range : tuple[int, int]
        Minimum and maximum temporal window length in frames. A new length
        is sampled uniformly at each iteration, acting as regularisation
        and making the model robust to variable trajectory lengths.
    dataset_size : int
        Number of subgraphs to serve per epoch, i.e. the value returned
        by __len__.
    position_std : float, optional
        Standard deviation of coordinate spans across Dt-length trajectory
        windows (both axes combined), used to scale node coordinates after
        per-subgraph centering. Typically estimated via
        GraphFromTrajectories.from_tracks. Default is 1.0.
    transform : callable, optional
        A transform following the PyG transform interface (Data -> Data)
        applied to each subgraph before statistics computation and scaling.
        Default is None.
    target : {"edges", "nodes", "global"}, optional
        Determines what y contains in each returned subgraph. "edges"
        returns per-edge ground-truth labels, "global" returns the
        graph-level class label. Default is "edges".
    sample_balanced : bool, optional
        If True, graphs are sampled uniformly across class labels rather
        than uniformly across graphs. Default is False.

    Methods
    -------
    compute_graph_statistics(graph)
        Compute local_scale and normalized_rg from raw node coordinates.
    center_and_scale_graph(graph)
        Center per-subgraph and scale by position_std.
    __len__()
        Return the dataset size.
    __getitem__(idx)
        Sample and return a random temporal subgraph.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        position_std: Optional[float] = None,
        connectivity_radius: Optional[float] = None,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
    ) -> None:
        self.Dt_range = Dt_range
        self.graph_dataset = [graph for graph in graph_dataset if len(graph.x) >= Dt_range[0]]
        self.dataset_size = dataset_size
        self.position_std = float(position_std) if position_std is not None else 1.0
        self.connectivity_radius = float(connectivity_radius) if connectivity_radius is not None else 1.0
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

    def __len__(self) -> int:
        """Return the number of subgraphs served per epoch.

        Returns
        -------
        int
            The dataset size passed at construction.
        """
        return self.dataset_size

    def compute_graph_statistics(self, graph: Data) -> Data:
        """Compute graph-level statistics from node coordinates.

        Parameters
        ----------
        graph : Data
            PyG Data object whose x attribute contains raw node coordinates.

        Returns
        -------
        Data
            The same graph with added attributes local_scale and normalized_rg.
        """
        coords = graph.x
        centered = coords - torch.mean(coords, dim=0)

        steps = torch.diff(coords, dim=0)
        local_scale = torch.norm(steps, dim=1).mean() + 1e-10

        T = coords.shape[0]
        rg = torch.norm(centered, dim=1).pow(2).mean().sqrt()
        expected_rg = torch.sqrt(local_scale ** 2 * T / 2)
        normalized_rg = rg / (expected_rg + 1e-10)

        graph.local_scale = local_scale.unsqueeze(0) / self.connectivity_radius
        graph.normalized_rg = normalized_rg.unsqueeze(0)
        graph.graph_features = torch.stack([local_scale, normalized_rg]).unsqueeze(0)
        return graph

    def center_and_scale_graph(self, graph: Data) -> Data:
        """Center and normalize node coordinates by mean step size.

        Parameters
        ----------
        graph : Data
            PyG Data object with x attribute and local_scale already computed
            by compute_graph_statistics.

        Returns
        -------
        Data
            A cloned Data object with centered and scaled node coordinates.
        """
        graph = graph.clone()
        coords = graph.x
        centered = coords - torch.mean(coords, dim=0)
        graph.x = centered / self.position_std
        return graph

    def __getitem__(self, idx: int) -> Data:
        """Sample a random temporal subgraph from the dataset.

        Selects a random graph, then samples a random time window of length
        drawn uniformly from Dt_range. Nodes and edges outside the window
        are masked out and edge indices are reindexed to be contiguous.
        Graph-level statistics (local_scale, normalized_rg) are computed
        from raw coordinates before centering and scaling.

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
                - distance : normalized distances, shape (E, 1)
                - y : target labels, shape depends on target mode
                - local_scale : mean step size before scaling, shape (1,)
                - normalized_rg : radius of gyration normalized by expected
                free diffusion radius, shape (1,)
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
            start_frame = np.random.choice(frames[frames <= (max_frame - self.Dt_range[0] + 1)].numpy())
            Dt = np.random.randint(self.Dt_range[0], self.Dt_range[1] + 1)
            end_frame = start_frame + Dt
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
        return_graph = self.compute_graph_statistics(return_graph)
        return_graph = self.center_and_scale_graph(return_graph)

        return return_graph
