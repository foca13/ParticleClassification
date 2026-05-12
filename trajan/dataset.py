from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
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
        Center node coordinates and normalize by local_scale.
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
        velocity_std: float = 1.0,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
    ) -> None:
        self.Dt_range = Dt_range
        self.graph_dataset = [graph for graph in graph_dataset if len(graph.x) >= Dt_range[0]]
        self.dataset_size = dataset_size
        self.velocity_std = velocity_std
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
        """Compute graph-level statistics from node velocity vectors.
        Parameters
        ----------
        graph : Data
            PyG Data object whose x attribute contains velocity vectors (N, 2).

        Returns
        -------
        Data
            The same graph with added attributes:
                - local_scale   : mean speed across all velocity nodes, shape (1,)
                - persistence   : mean cosine similarity between consecutive
                  velocity pairs, capturing directional persistence, shape (1,)
                - graph_features: [local_scale, persistence], shape (1, 2)
        """
        velocities = graph.x
        speeds = torch.norm(velocities, dim=1)
        local_scale = speeds.mean() + 1e-10

        if velocities.shape[0] > 1:
            v_norm = velocities / (speeds.unsqueeze(1) + 1e-10)
            persistence = (v_norm[:-1] * v_norm[1:]).sum(dim=1).mean()
        else:
            persistence = torch.zeros(1, device=velocities.device).squeeze()

        graph.local_scale = local_scale.unsqueeze(0)
        graph.persistence = persistence.unsqueeze(0)
        graph.graph_features = torch.stack([local_scale, persistence]).unsqueeze(0)

        return graph

    def center_and_scale_graph(self, graph: Data) -> Data:
        """Center and normalize velocity vectors by the training velocity std.

        Subtracts the mean velocity of the subgraph (removes drift) and divides
        by self.velocity_std (std of velocity increments from the training set),
        giving a consistent scale across all samples.

        Parameters
        ----------
        graph : Data
            PyG Data object with x containing raw velocity vectors.

        Returns
        -------
        Data
            A cloned Data object with centered and scaled velocity vectors.
        """
        graph = graph.clone()
        graph.x = (graph.x - graph.x.mean(dim=0)) / self.velocity_std
        return graph

    def __getitem__(self, _: int) -> Data:
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
                - x           : centered and speed-scaled velocities, shape (N, 2)
                - edge_index  : reindexed edge connectivity, shape (2, E)
                - edge_attr   : [rel. speed, norm. frame gap, cos angle], shape (E, 3)
                - y           : target labels, shape depends on target mode
                - local_scale : mean speed before scaling, shape (1,)
                - normalized_rg: mean velocity persistence (cosine of consecutive
                  velocity pairs), shape (1,)
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

        if self.target == TargetMode.EDGES:
            y = graph.y[edge_mask]
        elif self.target == TargetMode.GLOBAL:
            y = graph.graph_label
        else:
            raise NotImplementedError("Node-level targets are not yet supported.")

        return_graph = Data(
            x=graph.x[node_mask],
            edge_index=edge_index,
            edge_attr=graph.edge_attr[edge_mask],
            y=y,
        )

        if self.transform:
            return_graph = self.transform(return_graph)

        return_graph = self.compute_graph_statistics(return_graph)
        return_graph = self.center_and_scale_graph(return_graph)

        return return_graph


# ---------------------------------------------------------------------------
# Video-aware graph dataset
# ---------------------------------------------------------------------------

def build_video_graph_dataset(
    graph_builder,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    split_tracks: bool = True,
) -> List[Data]:
    """Build a graph dataset with a per-node ``crop_idx`` attribute.

    Mirrors ``GraphFromTrajectories.__call__`` but also attaches the index of
    the corresponding image crop (from ``video_tracks.csv``) to every velocity
    node.  Each velocity node represents the step ending at frame ``t``, so
    its crop is looked up as ``(set, label, frame_t)`` in the DataFrame.

    Parameters
    ----------
    graph_builder : GraphFromTrajectories
        A configured graph builder instance.
    df : pd.DataFrame
        ``video_tracks.csv`` DataFrame with columns
        ``[set, label, frame, crop_idx, ...]``.
    target_column : str, optional
        Column used as graph-level classification target.
    split_tracks : bool, optional
        If True, edges are within-trajectory only and the graph is split into
        per-trajectory subgraphs (default True).

    Returns
    -------
    list[Data]
        Same structure as ``GraphFromTrajectories.__call__`` output, with an
        additional ``crop_idx`` node attribute (shape ``(N,)``) on each Data.
    """
    # O(N) lookup: (set, label, frame) → crop_idx
    lookup = dict(zip(
        zip(df["set"], df["label"], df["frame"]),
        df["crop_idx"],
    ))

    if target_column is not None:
        df = df.copy()
        df["target"] = pd.Categorical(df[target_column]).codes

    graph_dataset: List[Data] = []

    for current_video in df["set"].unique():
        df_video = df[df["set"] == current_video]

        velocities, frames, labels, step_indices = graph_builder._compute_velocities(
            df_video, graph_builder.frame_rate
        )
        if len(velocities) == 0:
            continue

        sort_order = np.argsort(frames, kind="stable")
        velocities    = velocities[sort_order]
        frames        = frames[sort_order]
        labels        = labels[sort_order]
        step_indices  = step_indices[sort_order]

        # Each velocity node ends at (current_video, label, frame_later).
        crop_indices = np.array([
            lookup.get((current_video, int(labels[i]), int(frames[i])), -1)
            for i in range(len(velocities))
        ], dtype=np.int64)

        if split_tracks:
            edge_index, edge_attr = graph_builder.get_connectivity(velocities, frames, labels)
        else:
            edge_index, edge_attr = graph_builder.get_connectivity(velocities, frames)

        edge_gt = graph_builder.get_gt_connectivity(labels, step_indices, edge_index)

        graph_label = []
        if target_column is not None:
            graph_label = int(df_video.sort_values("frame")["target"].iloc[0])

        n_edges = len(edge_index)
        y = edge_gt[:, None].astype(np.float32) if n_edges > 0 else np.zeros((0, 1), dtype=np.float32)
        edge_index_t = (
            torch.tensor(edge_index.T, dtype=torch.long)
            if n_edges > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )

        graph = Data(
            x=torch.tensor(velocities, dtype=torch.float),
            edge_index=edge_index_t,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            frames=torch.tensor(frames, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float),
            graph_label=torch.tensor(graph_label, dtype=torch.int64),
            crop_idx=torch.tensor(crop_indices, dtype=torch.long),
        )

        if split_tracks:
            graph_dataset += graph_builder.get_subgraphs(graph)
        else:
            graph_dataset.append(graph)

    return graph_dataset


class VideoGraphDataset(torch.utils.data.Dataset):
    """Graph dataset that pairs each subgraph with a stack of image crops.

    Extends the ``GraphDataset`` sampling strategy to also return a
    ``(N_nodes, 1, H, W)`` tensor of per-node crops so that a model like
    ``ImageGraphConv`` can fuse appearance and motion information.

    Graphs must be built with :func:`build_video_graph_dataset` so that each
    node carries a ``crop_idx`` attribute pointing into the ``crops`` array.

    Parameters
    ----------
    graph_dataset : list[Data]
        List of PyG Data objects with ``crop_idx`` node attribute.
    crops : np.ndarray
        Float32 array of shape ``(N_total, H, W)`` — the full crop array from
        ``video_crops.npy``.
    Dt_range : tuple[int, int]
        Min and max temporal window length (frames) for subgraph sampling.
    dataset_size : int
        Number of samples returned per epoch.
    velocity_std : float
        Training-set velocity-increment std used to normalise node features.
    transform : callable, optional
        PyG-style transform applied to each subgraph before statistics.
    target : {"edges", "nodes", "global"}
        What ``y`` contains in each returned sample.
    sample_balanced : bool
        If True, graphs are sampled uniformly over class labels.
    """

    def __init__(
        self,
        graph_dataset: list,
        crops: np.ndarray,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        velocity_std: float = 1.0,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
    ) -> None:
        self.Dt_range = Dt_range
        self.graph_dataset = [g for g in graph_dataset if len(g.x) >= Dt_range[0]]
        # Store crops as (N_total, 1, H, W) float32, normalised to [0, 1].
        crops_f = crops.astype(np.float32)
        crops_f /= (crops_f.max() + 1e-8)
        self.crops = torch.tensor(crops_f[:, None, :, :], dtype=torch.float32)
        self.dataset_size = dataset_size
        self.velocity_std = velocity_std
        self.transform = transform
        self._target = None
        self.target = target
        self.labels = np.array([g.graph_label.item() for g in self.graph_dataset])
        self.sample_balanced = sample_balanced

    @property
    def target(self) -> TargetMode:
        return self._target

    @target.setter
    def target(self, value) -> None:
        self._target = TargetMode(value) if isinstance(value, str) else value

    def __len__(self) -> int:
        return self.dataset_size

    def compute_graph_statistics(self, graph: Data) -> Data:
        """Compute local_scale and persistence from raw velocity nodes."""
        velocities = graph.x
        speeds = torch.norm(velocities, dim=1)
        local_scale = speeds.mean() + 1e-10

        if velocities.shape[0] > 1:
            v_norm = velocities / (speeds.unsqueeze(1) + 1e-10)
            persistence = (v_norm[:-1] * v_norm[1:]).sum(dim=1).mean()
        else:
            persistence = torch.zeros(1, device=velocities.device).squeeze()

        graph.local_scale = local_scale.unsqueeze(0)
        graph.persistence = persistence.unsqueeze(0)
        graph.graph_features = torch.stack([local_scale, persistence]).unsqueeze(0)
        return graph

    def center_and_scale_graph(self, graph: Data) -> Data:
        """Centre and normalise velocity vectors by the training std."""
        graph = graph.clone()
        graph.x = (graph.x - graph.x.mean(dim=0)) / self.velocity_std
        return graph

    def __getitem__(self, _: int) -> Tuple[Data, torch.Tensor]:
        """Sample a random temporal subgraph and its per-node image crops.

        Returns
        -------
        graph : Data
            Subgraph with centred/scaled velocities and graph statistics.
        imgs : torch.Tensor
            Shape ``(N_nodes, 1, H, W)`` — one normalised crop per node.
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
        edge_index_sub = edge_index[:, edge_mask] - edge_index[:, edge_mask].min()

        if self.target == TargetMode.EDGES:
            y = graph.y[edge_mask]
        elif self.target == TargetMode.GLOBAL:
            y = graph.graph_label
        else:
            raise NotImplementedError("Node-level targets are not yet supported.")

        return_graph = Data(
            x=graph.x[node_mask],
            edge_index=edge_index_sub,
            edge_attr=graph.edge_attr[edge_mask],
            y=y,
        )

        # Crops are retrieved before the transform so they can be spatially
        # augmented with the same random parameters as the velocity vectors.
        imgs = self.crops[graph.crop_idx[node_mask]]  # (N, 1, H, W)

        if self.transform:
            return_graph, imgs = self.transform(return_graph, imgs)

        return_graph = self.compute_graph_statistics(return_graph)
        return_graph = self.center_and_scale_graph(return_graph)

        return return_graph, imgs
