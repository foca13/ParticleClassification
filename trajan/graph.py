from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from .data import TracksDataFrame


class GraphFromTrajectories:
    """Graph representation of particle trajectories using velocity nodes.

    Each node is a velocity step (vx, vy) computed between two consecutive
    detections of the same particle. Edges connect velocity nodes that are
    within max_frame_gap frames apart and within max_velocity_diff in L2
    velocity space.

    Parameters
    ----------
    max_velocity_diff : float
        Maximum L2 norm of the velocity difference between two nodes for an
        edge to be created.
    max_frame_gap : int
        Maximum frame gap between two velocity nodes for an edge to be created.
    frame_rate : float, optional
        Frame rate used to convert position differences to velocities (px/s).
        Default is 1.

    Methods
    -------
    estimate_max_velocity_diff(velocity_changes, sigma_deviation)
        Estimate a velocity-difference threshold from observed increments.
    from_tracks(df, max_frame_gap, sigma_deviation, frame_rate)
        Convenience constructor that estimates parameters from training data.
    get_subgraphs(graph)
        Split a graph into its connected components.
    get_connectivity(velocities, frame_indices, labels)
        Compute candidate edges and edge features between velocity nodes.
    get_gt_connectivity(labels, step_indices, edge_index)
        Compute ground-truth edge labels from particle and step identifiers.
    __call__(df, target_column, split_tracks)
        Build a list of PyG Data objects from a TracksDataFrame.
    """

    def __init__(
        self,
        max_velocity_diff: float,
        max_frame_gap: int,
        frame_rate: float = 1.0,
    ) -> None:
        self.max_velocity_diff = max_velocity_diff
        self.max_frame_gap = max_frame_gap
        self.frame_rate = frame_rate

    @staticmethod
    def _compute_velocities(
        df_video: pd.DataFrame,
        frame_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute velocity nodes from per-particle position sequences.

        For each particle with N detections, produces N-1 velocity vectors.
        Each velocity node is assigned the frame of its later detection and a
        0-based step index within the particle's trajectory.

        Returns
        -------
        tuple of four np.ndarray
            velocities (N, 2), frames (N,), labels (N,), step_indices (N,)
        """
        vx_list, vy_list, frame_list, label_list, step_list = [], [], [], [], []

        for label in df_video["label"].unique():
            particle = df_video[df_video["label"] == label].sort_values("frame")
            frames_p = particle["frame"].to_numpy()
            x = particle["x"].to_numpy()
            y = particle["y"].to_numpy()

            for i in range(1, len(frames_p)):
                dt = (frames_p[i] - frames_p[i - 1]) / frame_rate
                vx_list.append((x[i] - x[i - 1]) / dt)
                vy_list.append((y[i] - y[i - 1]) / dt)
                frame_list.append(frames_p[i])
                label_list.append(label)
                step_list.append(i - 1)

        if not vx_list:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )

        return (
            np.column_stack([vx_list, vy_list]).astype(np.float32),
            np.array(frame_list, dtype=np.int64),
            np.array(label_list, dtype=np.int64),
            np.array(step_list, dtype=np.int64),
        )

    @staticmethod
    def estimate_max_velocity_diff(
        velocity_changes: np.ndarray,
        sigma_deviation: float = 3,
    ) -> float:
        """Estimate a velocity-difference threshold from observed increments.

        Parameters
        ----------
        velocity_changes : np.ndarray
            1D array of velocity increment magnitudes |v_{i+1} - v_i|.
        sigma_deviation : float, optional
            Number of standard deviations above the mean. Default is 3.

        Returns
        -------
        float
            mean + sigma_deviation * std of the velocity change magnitudes.
        """
        return float(np.mean(velocity_changes) + sigma_deviation * np.std(velocity_changes))

    @classmethod
    def from_tracks(
        cls,
        df: TracksDataFrame,
        max_frame_gap: int,
        frame_rate: float = 1.0,
        sigma_deviation: float = 3,
    ) -> tuple["GraphFromTrajectories", float]:
        """Convenience constructor that estimates parameters from training data.

        Collects all frame-to-frame velocity changes across every trajectory,
        estimates max_velocity_diff as mean + sigma_deviation * std, and
        returns the std of those increments for downstream normalization.

        Parameters
        ----------
        df : TracksDataFrame
            Training tracking data.
        max_frame_gap : int
            Maximum frame gap between two connected velocity nodes.
        frame_rate : float, optional
            Frame rate used to compute velocities. Default is 1.
        sigma_deviation : float, optional
            Passed to estimate_max_velocity_diff. Default is 3.

        Returns
        -------
        tuple[GraphFromTrajectories, float]
            The instantiated graph builder and the std of velocity increments.
        """
        all_velocity_changes = []

        for video in df["set"].unique():
            df_video = df[df["set"] == video]
            for label in df_video["label"].unique():
                particle = df_video[df_video["label"] == label].sort_values("frame")
                frames_p = particle["frame"].to_numpy()
                x = particle["x"].to_numpy()
                y = particle["y"].to_numpy()

                if len(frames_p) < 3:
                    continue

                velocities = []
                for i in range(1, len(frames_p)):
                    dt = (frames_p[i] - frames_p[i - 1]) / frame_rate
                    velocities.append(np.array([(x[i] - x[i - 1]) / dt, (y[i] - y[i - 1]) / dt]))

                for i in range(1, len(velocities)):
                    all_velocity_changes.append(float(np.linalg.norm(velocities[i] - velocities[i - 1])))

        velocity_changes = np.array(all_velocity_changes) if all_velocity_changes else np.array([0.0])
        velocity_std = float(np.std(velocity_changes))
        max_velocity_diff = cls.estimate_max_velocity_diff(velocity_changes, sigma_deviation)

        return cls(
            max_velocity_diff=max_velocity_diff,
            max_frame_gap=max_frame_gap,
            frame_rate=frame_rate,
        ), velocity_std

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
            mask = torch.from_numpy(component == c).to(graph.edge_index.device, torch.bool)
            subgraphs.append(graph.subgraph(mask))
        return subgraphs

    def get_connectivity(
        self,
        velocities: np.ndarray,
        frame_indices: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute candidate edges and edge features between velocity nodes.

        Creates an edge between nodes i and j if their frame gap is within
        max_frame_gap and their velocity L2 difference is below
        max_velocity_diff. If labels is provided, only same-label pairs are
        considered.

        Edge features:
            - log speed ratio    : log((|v_j| + ε) / (|v_i| + ε))
            - mean speed         : (|v_i| + |v_j|) / (2 * max_velocity_diff)
            - velocity diff norm : |v_i - v_j| / max_velocity_diff
            - normalized frame gap: frame_gap / max_frame_gap
            - cosine angle       : cos(v_i, v_j)

        Parameters
        ----------
        velocities : np.ndarray, shape (N, 2)
        frame_indices : np.ndarray, shape (N,), sorted ascending
        labels : np.ndarray, shape (N,), optional
            If provided, only same-label pairs are connected.

        Returns
        -------
        edges : np.ndarray, shape (E, 2)
        edge_features : np.ndarray, shape (E, 5)
        """
        edges = []
        edge_features = []
        num_nodes = len(velocities)
        speeds = np.linalg.norm(velocities, axis=1)

        for node_idx in range(num_nodes):
            node_frame = frame_indices[node_idx]
            node_label = labels[node_idx] if labels is not None else None

            for neighbor_idx in range(node_idx + 1, num_nodes):
                neighbor_frame = frame_indices[neighbor_idx]
                frame_gap = int(neighbor_frame - node_frame)

                if frame_gap <= 0:
                    continue
                if frame_gap > self.max_frame_gap:
                    break
                if labels is not None and labels[neighbor_idx] != node_label:
                    continue

                velocity_diff = np.linalg.norm(velocities[node_idx] - velocities[neighbor_idx])
                if velocity_diff >= self.max_velocity_diff:
                    continue

                speed_i = speeds[node_idx]
                speed_j = speeds[neighbor_idx]
                log_speed_ratio = float(np.log((speed_j + 1e-6) / (speed_i + 1e-6)))
                mean_speed = float((speed_i + speed_j) / (2 * self.max_velocity_diff))
                velocity_diff_norm = float(velocity_diff / self.max_velocity_diff)
                cos_angle = float(
                    np.dot(velocities[node_idx], velocities[neighbor_idx])
                    / (speed_i * speed_j + 1e-10)
                )

                edges.append([node_idx, neighbor_idx])
                edge_features.append([
                    log_speed_ratio,
                    mean_speed,
                    velocity_diff_norm,
                    frame_gap / self.max_frame_gap,
                    cos_angle,
                ])

        if not edges:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)

        return np.array(edges, dtype=np.int64), np.array(edge_features, dtype=np.float32)

    def get_gt_connectivity(
        self,
        labels: np.ndarray,
        step_indices: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """Compute ground-truth edge labels for velocity nodes.

        An edge is a ground-truth connection if the two nodes belong to the
        same particle and are consecutive velocity steps (step difference = 1).

        Parameters
        ----------
        labels : np.ndarray, shape (N,)
            Particle identifier for each velocity node.
        step_indices : np.ndarray, shape (N,)
            0-based step index within each particle's trajectory.
        edge_index : np.ndarray, shape (E, 2)

        Returns
        -------
        np.ndarray, shape (E,), dtype bool
        """
        if len(edge_index) == 0:
            return np.zeros(0, dtype=bool)

        src, tgt = edge_index[:, 0], edge_index[:, 1]
        return (labels[src] == labels[tgt]) & (step_indices[tgt] == step_indices[src] + 1)

    def __call__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        split_tracks: bool = True,
    ) -> list[Data]:
        """Build a list of PyG Data objects from a TracksDataFrame.

        For each video, velocity nodes are computed from per-particle position
        sequences and edges are created based on velocity similarity and frame
        proximity. When split_tracks is True, only within-trajectory edges are
        allowed and the graph is split into one subgraph per trajectory.

        Parameters
        ----------
        df : pd.DataFrame
            Tracking data with columns ["set", "frame", "x", "y", "label"].
        target_column : str, optional
            Column for graph-level classification target. Default is None.
        split_tracks : bool, optional
            If True, edges are only created within trajectories and the result
            is split into connected components (one per trajectory).
            Default is True.

        Returns
        -------
        list[Data]
            Each Data object contains:
                - x          : velocity node features (vx, vy), shape (N, 2)
                - edge_index : edge connectivity, shape (2, E)
                - edge_attr  : [log speed ratio, mean speed, vel. diff norm, norm. frame gap, cos angle], shape (E, 5)
                - frames     : frame indices of velocity nodes, shape (N,)
                - y          : ground-truth edge labels, shape (E, 1)
                - graph_label: graph-level class label, shape ()
        """
        graph_dataset = []

        if target_column is not None:
            df = df.copy()
            df["target"] = pd.Categorical(df[target_column]).codes

        for current_video in df["set"].unique():
            df_video = df[df["set"] == current_video]

            velocities, frames, labels, step_indices = self._compute_velocities(df_video, self.frame_rate)

            if len(velocities) == 0:
                continue

            sort_order = np.argsort(frames, kind="stable")
            velocities = velocities[sort_order]
            frames = frames[sort_order]
            labels = labels[sort_order]
            step_indices = step_indices[sort_order]

            if split_tracks:
                edge_index, edge_attr = self.get_connectivity(velocities, frames, labels)
            else:
                edge_index, edge_attr = self.get_connectivity(velocities, frames)

            edge_gt = self.get_gt_connectivity(labels, step_indices, edge_index)

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
            )

            if split_tracks:
                graph_dataset += self.get_subgraphs(graph)
            else:
                graph_dataset.append(graph)

        return graph_dataset
