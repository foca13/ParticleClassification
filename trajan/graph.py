from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from .data import TracksDataFrame


class GraphFromTrajectories:
    """Graph representation of particle trajectories for classification.

    Each node corresponds to a single particle detection at a specific frame.
    Edges connect detections that are within a spatial radius and a maximum
    frame distance of each other, representing candidate temporal links.
    Ground-truth edges connect only consecutive detections of the same
    particle.

    Parameters
    ----------
    connectivity_radius : float
        Maximum spatial distance between two detections for an edge to
        be created.
    max_frame_distance : int
        Maximum number of frames between two detections for an edge to
        be created.

    Methods
    -------
    estimate_connectivity_radius(displacements, sigma_deviation, scaling)
        Estimate a connectivity radius from observed displacements.
    estimate_trajectory_span_std(df, Dt)
        Estimate a single position scale from trajectory span data.
    from_tracks(df, Dt, max_frame_distance, sigma_deviation, scaling)
        Convenience constructor that estimates parameters from the data.
    get_subgraphs(graph)
        Split a graph into its connected components.
    get_connectivity(positions, frame_indices, labels)
        Compute candidate edges and edge features for a set of detections.
    get_gt_connectivity(labels, edge_index, times)
        Compute ground-truth edge labels from particle identifiers.
    __call__(df, target_column, split_tracks)
        Build a list of PyG Data objects from a TracksDataFrame.
    """

    def __init__(
        self,
        connectivity_radius: float,
        max_frame_distance: int,
    ) -> None:
        """Initialize the graph builder.

        Parameters
        ----------
        connectivity_radius : float
            Maximum spatial distance between two detections for an edge
            to be created.
        max_frame_distance : int
            Maximum number of frames between two detections for an edge
            to be created.
        """
        self.connectivity_radius = connectivity_radius
        self.max_frame_distance = max_frame_distance

    @staticmethod
    def estimate_connectivity_radius(
        displacements: np.ndarray,
        sigma_deviation: float = 3,
        scaling: float = 1,
    ) -> float:
        """Estimate a connectivity radius from observed displacements.

        Computes an upper bound on expected displacements as the mean plus
        `sigma_deviation` standard deviations, then rounds up to the nearest
        order of magnitude for a clean threshold.

        Parameters
        ----------
        displacements : np.ndarray
            1D array of per-step displacement magnitudes, as returned by
            TracksDataFrame.compute_displacements().
        sigma_deviation : float, optional
            Number of standard deviations above the mean to use as the
            upper bound. Default is 3.
        scaling : float, optional
            Additional scaling factor applied to the upper bound before
            rounding. Default is 1.

        Returns
        -------
        float
            Estimated connectivity radius, rounded up to the nearest
            order of magnitude.
        """
        upper_displacement = scaling * (np.mean(displacements) + sigma_deviation * np.std(displacements))
        order_of_magnitude = 10 ** np.floor(np.log10(upper_displacement))
        connectivity_radius = np.ceil(upper_displacement / order_of_magnitude) * order_of_magnitude
        return connectivity_radius

    @staticmethod
    def estimate_trajectory_span_std(df: TracksDataFrame, Dt: int) -> float:
        """Estimate a single position scale from coordinate spans within Dt-length windows.

        For each particle trajectory, slides a window of length Dt and computes
        the coordinate range (max - min) along x and y. Treats both axes as
        samples of the same distribution (since trajectory orientation is
        arbitrary) and returns their combined standard deviation as a scalar.

        Parameters
        ----------
        df : TracksDataFrame
            Tracking data containing at least "x", "y", "set", "label", "frame".
        Dt : int
            Window length in frames.

        Returns
        -------
        float
            Standard deviation of all coordinate spans across both axes and
            all Dt-length windows.
        """
        spans = []
        for video in df["set"].unique():
            df_video = df[df["set"] == video]
            for particle in df_video["label"].unique():
                particle_data = df_video[df_video["label"] == particle].sort_values("frame")
                coords = particle_data[["x", "y"]].to_numpy()
                for i in range(len(coords) - Dt + 1):
                    span = np.ptp(coords[i:i + Dt], axis=0)
                    spans.append(span)
        if not spans:
            return 1.0
        spans = np.array(spans)
        return float(spans.std())

    @classmethod
    def from_tracks(
        cls,
        df: TracksDataFrame,
        Dt: int,
        max_frame_distance: int,
        sigma_deviation: float = 3,
        scaling: float = 1,
    ) -> tuple["GraphFromTrajectories", float]:
        """Convenience constructor that estimates parameters from the data.

        Estimates the connectivity radius from observed displacements and
        position statistics from Dt-length window spans, then instantiates
        the graph builder with the estimated connectivity radius.

        Parameters
        ----------
        df : TracksDataFrame
            Tracking data used to estimate parameters.
        Dt : int
            Length of the time window in frames, used to estimate the
            position scale.
        max_frame_distance : int
            Maximum number of frames between two connected detections.
        sigma_deviation : float, optional
            Passed to estimate_connectivity_radius. Default is 3.
        scaling : float, optional
            Passed to estimate_connectivity_radius. Default is 1.

        Returns
        -------
        tuple[GraphFromTrajectories, np.ndarray]
            The instantiated graph builder and the position std (shape (2,))
            estimated from Dt-length window spans, for coordinate scaling.
        """
        displacements = df.compute_displacements()
        connectivity_radius = cls.estimate_connectivity_radius(displacements, sigma_deviation, scaling)
        trajectory_span_std = cls.estimate_trajectory_span_std(df, Dt)
        return cls(connectivity_radius=connectivity_radius, max_frame_distance=max_frame_distance), trajectory_span_std

    @staticmethod
    def get_subgraphs(graph: Data) -> list[Data]:
        """Split a graph into its connected components.

        Parameters
        ----------
        graph : Data
            A PyG Data object to split.

        Returns
        -------
        list[Data]
            A list of PyG Data objects, one per connected component.
        """
        adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes)
        num_components, component = sp.csgraph.connected_components(adj, directed=False)
        subgraphs = []
        for c in range(num_components):
            subgraph_mask = torch.from_numpy(component == c).to(graph.edge_index.device, torch.bool)
            subgraphs.append(graph.subgraph(subgraph_mask))
        return subgraphs

    def get_connectivity(
        self,
        positions: np.ndarray,
        frame_indices: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute candidate edges and edge features for a set of detections.

        Iterates over all pairs of detections and creates an edge if they
        are within `connectivity_radius` spatially and `max_frame_distance`
        frames apart. If `labels` is provided, only detections sharing the
        same label are connected (used to build ground-truth-only graphs).

        Edge features are:
            - normalised distance (distance / connectivity_radius)
            - normalised frame gap (frame_gap / max_frame_distance)
            - a motion energy proxy (distance² / normalised frame gap)

        Parameters
        ----------
        positions : np.ndarray
            Array of shape (N, 2) containing the (x, y) coordinates of
            each detection.
        frame_indices : np.ndarray
            Array of shape (N,) containing the frame index of each detection.
            Must be sorted in ascending order.
        labels : np.ndarray, optional
            Array of shape (N,) containing particle identifiers. If provided,
            only same-label pairs are considered. Default is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            edges : np.ndarray of shape (E, 2)
                Source and target node indices for each edge.
            edge_features : np.ndarray of shape (E, 3)
                Feature vector for each edge.
        """
        edges = []
        edge_features = []
        num_nodes = len(positions)

        for node_idx in range(num_nodes):
            node_frame = frame_indices[node_idx]
            if labels is not None:
                node_label = labels[node_idx]

            for neighbor_idx in range(node_idx + 1, num_nodes):
                neighbor_frame = frame_indices[neighbor_idx]
                frame_gap = neighbor_frame - node_frame

                if labels is not None:
                    if labels[neighbor_idx] != node_label:
                        continue
                if frame_gap <= 0:
                    continue
                if frame_gap > self.max_frame_distance:
                    break
                distance = np.linalg.norm(positions[node_idx] - positions[neighbor_idx])

                if distance < self.connectivity_radius:
                    edges.append([node_idx, neighbor_idx])
                    norm_distance = distance / self.connectivity_radius
                    norm_frame_gap = frame_gap / self.max_frame_distance
                    edge_features.append([
                        norm_distance,
                        norm_frame_gap,
                        (norm_distance ** 2) / norm_frame_gap,
                    ])

        edges = np.array(edges, dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32)

        return edges, edge_features

    def get_gt_connectivity(
        self,
        labels: np.ndarray,
        edge_index: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute ground-truth edge labels from particle identifiers.

        An edge is considered a ground-truth connection if and only if its
        source and target are consecutive detections of the same particle,
        i.e. there is no earlier detection of that particle between them.

        Parameters
        ----------
        labels : np.ndarray
            Array of shape (N,) containing particle identifiers for each node.
        edge_index : np.ndarray
            Array of shape (E, 2) containing source and target node indices.
        times : np.ndarray
            Array of shape (N,) containing frame indices for each node.

        Returns
        -------
        np.ndarray
            Boolean array of shape (E,) where True indicates a ground-truth
            connection between consecutive detections of the same particle.
        """
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        gt_connectivity = np.zeros(len(src), dtype=bool)

        for pid in np.unique(labels):
            particle_nodes = np.where(labels == pid)[0]
            sorted_idx = np.argsort(times[particle_nodes])
            sorted_nodes = particle_nodes[sorted_idx]
            if len(sorted_nodes) < 2:
                continue
            consecutive_pairs = set(zip(sorted_nodes[:-1], sorted_nodes[1:]))
            for i, (s, t) in enumerate(zip(src, tgt)):
                if (s, t) in consecutive_pairs:
                    gt_connectivity[i] = True

        return gt_connectivity

    def __call__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        split_tracks: bool = True,
    ) -> list[Data]:
        """Build a list of PyG Data objects from a TracksDataFrame.

        Each unique video in the DataFrame is converted into a PyG Data
        object. Nodes represent detections and edges represent candidate
        temporal links, with ground-truth labels indicating which edges
        correspond to true consecutive detections of the same particle.

        Parameters
        ----------
        df : pd.DataFrame
            Tracking data containing at least the columns
            ["set", "frame", "x", "y", "label"].
        target_column : str, optional
            Column to use as a graph-level classification target. Its
            values are encoded as integer codes. Default is None.
        split_tracks : bool, optional
            If True, edges are only created between detections sharing
            the same label, and the resulting graph is split into
            connected components. Default is False.

        Returns
        -------
        list[Data]
            A list of PyG Data objects, one per video (or per connected
            component if split_tracks is True). Each object contains:
                - x : node coordinates, shape (N, 2)
                - edge_index : edge connectivity, shape (2, E)
                - edge_attr : edge feature vectors, shape (E, 3)
                - distance : normalised distances, shape (E, 1)
                - frames : frame indices, shape (N,)
                - y : ground-truth edge labels, shape (E, 1)
                - graph_label : graph-level class label, shape ()
        """
        graph_dataset = []

        if target_column is not None:
            df = df.copy()
            df["target"] = pd.Categorical(df[target_column]).codes

        for current_video in df["set"].unique():
            df_video = df[df["set"] == current_video]
            df_video = df_video.sort_values(by=["frame"]).reset_index(drop=True)

            positions = df_video[["x", "y"]].to_numpy()
            node_labels = df_video["label"].to_numpy()
            frames = df_video["frame"].to_numpy()

            if split_tracks:
                edge_index, edge_attr = self.get_connectivity(positions, frames, node_labels)
            else:
                edge_index, edge_attr = self.get_connectivity(positions, frames)
            edge_gt = self.get_gt_connectivity(node_labels, edge_index, frames)

            graph_label = df_video["target"][0] if target_column is not None else []

            graph = Data(
                x=torch.tensor(positions, dtype=torch.float),
                edge_index=torch.tensor(edge_index.T, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                distance=torch.tensor(edge_attr[:, :1], dtype=torch.float),
                frames=torch.tensor(frames, dtype=torch.float),
                y=torch.tensor(edge_gt[:, None], dtype=torch.float),
                graph_label=torch.tensor(graph_label, dtype=torch.int64),
            )

            if split_tracks:
                graph_dataset += self.get_subgraphs(graph)
            else:
                graph_dataset.append(graph)

        return graph_dataset
