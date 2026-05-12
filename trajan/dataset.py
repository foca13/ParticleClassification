import inspect
from enum import Enum
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class TargetMode(Enum):
    EDGES = "edges"
    NODES = "nodes"
    GLOBAL = "global"


# ---------------------------------------------------------------------------
# Crop normalisation utilities
# ---------------------------------------------------------------------------

def normalize_crops(
    crops: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Normalise single-channel crop arrays to [0, 1].

    Parameters
    ----------
    crops : np.ndarray
        Shape ``(N, H, W)`` or ``(N, 1, H, W)``, dtype float or uint.
    mode : {"global", "per_crop"}
        ``"global"``   — 1st/99th-percentile min-max over the full dataset.
                         Preserves relative brightness across particles.
        ``"per_crop"`` — 2nd/98th-percentile min-max per crop.
                         Maximises local contrast; removes inter-particle
                         brightness variation.

    Returns
    -------
    np.ndarray
        Same shape as input, dtype float32, values clipped to [0, 1].
    """
    crops = np.asarray(crops, dtype=np.float32)
    if mode == "global":
        lo = np.percentile(crops, 1)
        hi = np.percentile(crops, 99)
        denom = hi - lo if hi > lo else 1.0
        return np.clip((crops - lo) / denom, 0.0, 1.0)
    elif mode == "per_crop":
        original_shape = crops.shape
        flat = crops.reshape(len(crops), -1)
        lo = np.percentile(flat, 2, axis=1, keepdims=True)
        hi = np.percentile(flat, 98, axis=1, keepdims=True)
        denom = np.where(hi > lo, hi - lo, 1.0)
        normalised = np.clip((flat - lo) / denom, 0.0, 1.0)
        return normalised.reshape(original_shape)
    else:
        raise ValueError(f"Unknown crop normalisation mode: {mode!r}. Expected 'global' or 'per_crop'.")


# ---------------------------------------------------------------------------
# Shared sampling base class
# ---------------------------------------------------------------------------

class _GraphDatasetBase(torch.utils.data.Dataset):
    """Shared temporal-sampling infrastructure for trajectory graph datasets.

    Handles graph selection, time-window extraction, and edge/node masking.
    Subclasses implement ``_process_subgraph`` to apply mode-specific
    transforms, feature computation, and normalisation.

    Parameters
    ----------
    graph_dataset : list[Data]
        PyG Data objects, one per trajectory (or connected component).
        Graphs with fewer nodes than ``Dt_range[0]`` are discarded.
    Dt_range : tuple[int, int]
        Min and max temporal window length (frames). A new length is drawn
        uniformly at each step, acting as a regularisation.
    dataset_size : int
        Number of subgraphs served per epoch (value of ``__len__``).
    transform : callable, optional
        PyG-style transform (Data → Data) applied inside ``_process_subgraph``
        before statistics and normalisation. Default is None.
    target : {"edges", "nodes", "global"}, optional
        What ``y`` contains in the returned graph. Default is ``"global"``.
    sample_balanced : bool, optional
        If True, graphs are sampled uniformly over class labels. Default True.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
    ) -> None:
        self.Dt_range = Dt_range
        self.graph_dataset = [g for g in graph_dataset if len(g.x) >= Dt_range[0]]
        self.dataset_size = dataset_size
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

    def _sample_subgraph(self) -> Data:
        """Select a random graph and extract a random temporal window from it.

        Returns a Data object with ``x``, ``edge_index``, ``edge_attr``, and
        ``y`` set. Node features are still in raw (un-normalised) form.
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
            start_frame = np.random.choice(
                frames[frames <= (max_frame - self.Dt_range[0] + 1)].numpy()
            )
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

        subgraph = Data(
            x=graph.x[node_mask],
            edge_index=edge_index_sub,
            edge_attr=graph.edge_attr[edge_mask],
            y=y,
        )

        # Forward any extra node-level attributes (e.g. crop_idx, crop_idx_next)
        # so that subclasses can access them without knowing about them here.
        _standard = {"x", "edge_index", "edge_attr", "y", "frames", "graph_label"}
        for key, val in graph:
            if (
                key not in _standard
                and isinstance(val, torch.Tensor)
                and val.size(0) == graph.num_nodes
            ):
                subgraph[key] = val[node_mask]

        return subgraph

    def _process_subgraph(self, graph: Data) -> Data:
        raise NotImplementedError

    def __getitem__(self, _: int) -> Data:
        return self._process_subgraph(self._sample_subgraph())


# ---------------------------------------------------------------------------
# Velocity-node dataset
# ---------------------------------------------------------------------------

class VelocityGraphDataset(_GraphDatasetBase):
    """Graph dataset for velocity-node graphs (VelocityGraphFromTrajectories).

    Node features (vx, vy) are standardised by the training-set velocity-
    increment std but NOT centred: velocities are already location-invariant,
    so centering would erase directional information. The sole graph-level
    feature is directional persistence (mean cosine similarity of consecutive
    velocity pairs), which is scale-invariant and computed before normalisation.

    Parameters
    ----------
    graph_dataset : list[Data]
        Output of ``VelocityGraphFromTrajectories.__call__``.
    Dt_range : tuple[int, int]
        Min/max temporal window length.
    dataset_size : int
        Samples served per epoch.
    velocity_std : float
        Std of velocity increments from ``VelocityGraphFromTrajectories.from_tracks``.
    transform : callable, optional
        PyG-style transform applied before persistence computation.
    target : {"edges", "nodes", "global"}
        What ``y`` contains.
    sample_balanced : bool
        Sample uniformly over class labels if True.
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
        super().__init__(graph_dataset, Dt_range, dataset_size, transform, target, sample_balanced)
        self.velocity_std = velocity_std

    def _process_subgraph(self, graph: Data) -> Data:
        if self.transform:
            graph = self.transform(graph)

        # Persistence from raw (un-normalised) velocities — scale-invariant
        velocities = graph.x
        speeds = torch.norm(velocities, dim=1)
        if velocities.shape[0] > 1:
            v_norm = velocities / (speeds.unsqueeze(1) + 1e-10)
            persistence = (v_norm[:-1] * v_norm[1:]).sum(dim=1).mean()
        else:
            persistence = torch.zeros(1, device=velocities.device).squeeze()

        graph.persistence = persistence.unsqueeze(0)
        graph.graph_features = persistence.unsqueeze(0).unsqueeze(0)  # (1, 1)

        # Standardise by training-set velocity std; no centring
        graph.x = graph.x / self.velocity_std

        return graph


# ---------------------------------------------------------------------------
# Position-node dataset
# ---------------------------------------------------------------------------

class PositionGraphDataset(_GraphDatasetBase):
    """Graph dataset for position-node graphs (PositionGraphFromTrajectories).

    Each sampled subgraph is normalised in a way that is unit-independent and
    Dt-independent:

    1. Transforms (rotation / flip) are applied to raw coordinates.
    2. Graph-level features are computed from the transformed but un-normalised
       positions so that they retain physical meaning for the classifier head:
         - ``local_scale``: mean per-step displacement magnitude (∝ √D).
         - ``normalized_rg``: radius of gyration / expected RG for a random walk.
    3. Positions are divided by the std of per-step displacement magnitudes of
       the subgraph (unit-independent local scale).
    4. The subgraph is centred by subtracting the per-subgraph mean position.

    Parameters
    ----------
    graph_dataset : list[Data]
        Output of ``PositionGraphFromTrajectories.__call__``.
    Dt_range : tuple[int, int]
        Min/max temporal window length.
    dataset_size : int
        Samples served per epoch.
    transform : callable, optional
        PyG-style transform applied before stats and normalisation.
    target : {"edges", "nodes", "global"}
        What ``y`` contains.
    sample_balanced : bool
        Sample uniformly over class labels if True.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
    ) -> None:
        super().__init__(graph_dataset, Dt_range, dataset_size, transform, target, sample_balanced)

    def _process_subgraph(self, graph: Data) -> Data:
        if self.transform:
            graph = self.transform(graph)

        # --- graph-level features from raw (transformed) positions ---
        coords = graph.x
        steps = torch.diff(coords, dim=0)
        step_sizes = torch.norm(steps, dim=1)
        local_scale = step_sizes.mean() + 1e-10

        T = coords.shape[0]
        centered_raw = coords - coords.mean(dim=0)
        rg = torch.norm(centered_raw, dim=1).pow(2).mean().sqrt()
        expected_rg = torch.sqrt(local_scale ** 2 * T / 2)
        normalized_rg = rg / (expected_rg + 1e-10)

        graph.local_scale = local_scale.unsqueeze(0)
        graph.normalized_rg = normalized_rg.unsqueeze(0)
        graph.graph_features = torch.stack([local_scale, normalized_rg]).unsqueeze(0)  # (1, 2)

        # --- normalise: divide by local step std, then centre ---
        # fall back to mean step size when std is undefined (≤1 step) or zero
        step_std = step_sizes.std() if step_sizes.numel() > 1 else local_scale - 1e-10
        step_std = step_std + 1e-10
        graph.x = graph.x / step_std
        graph.x = graph.x - graph.x.mean(dim=0)

        return graph


# ---------------------------------------------------------------------------
# Video-aware graph dataset (with per-node image crops)
# ---------------------------------------------------------------------------

def _build_velocity_video_graphs(
    graph_builder,
    df: pd.DataFrame,
    crops_arr: np.ndarray,
    target_column: Optional[str],
    split_tracks: bool,
) -> Tuple[List[Data], np.ndarray]:
    """Build velocity-node graphs and pre-compute per-node frame-difference crops.

    For each velocity node (start frame → end frame) the difference crop
    ``crops_arr[end_idx] − crops_arr[start_idx]`` is stored in a new array.
    Each node carries a single ``crop_idx`` into that diff-crops array.

    Returns
    -------
    graph_dataset : list[Data]
    diff_crops : np.ndarray, shape (M, H, W), dtype float32
    """
    from trajan.graph import get_subgraphs

    lookup = dict(zip(
        zip(df["set"], df["label"], df["frame"]),
        df["crop_idx"],
    ))

    if target_column is not None:
        df = df.copy()
        df["target"] = pd.Categorical(df[target_column]).codes

    h, w = crops_arr.shape[-2], crops_arr.shape[-1]
    graph_dataset: List[Data] = []
    diff_crops_list: List[np.ndarray] = []
    global_offset = 0

    for current_video in df["set"].unique():
        df_video = df[df["set"] == current_video]

        velocities, frames, labels, step_indices = graph_builder._compute_velocities(
            df_video, graph_builder.frame_rate
        )
        if len(velocities) == 0:
            continue

        sort_order = np.argsort(frames, kind="stable")
        velocities   = velocities[sort_order]
        frames       = frames[sort_order]
        labels       = labels[sort_order]
        step_indices = step_indices[sort_order]

        # Build (label, step) → frame mapping for start-frame lookup
        step_to_frame: dict = {}
        for lbl in df_video["label"].unique():
            p = df_video[df_video["label"] == lbl].sort_values("frame")
            for step, f in enumerate(p["frame"].tolist()):
                step_to_frame[(int(lbl), step)] = int(f)

        n_nodes = len(velocities)
        for i in range(n_nodes):
            start_frame = step_to_frame.get((int(labels[i]), int(step_indices[i])), -1)
            end_frame   = int(frames[i])
            start_idx   = lookup.get((current_video, int(labels[i]), start_frame), -1)
            end_idx     = lookup.get((current_video, int(labels[i]), end_frame), -1)
            if start_idx >= 0 and end_idx >= 0:
                diff = crops_arr[end_idx].astype(np.float32) - crops_arr[start_idx].astype(np.float32)
            else:
                diff = np.zeros((h, w), dtype=np.float32)
            diff_crops_list.append(diff)

        crop_indices = np.arange(global_offset, global_offset + n_nodes, dtype=np.int64)
        global_offset += n_nodes

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
            graph_dataset += get_subgraphs(graph)
        else:
            graph_dataset.append(graph)

    diff_crops = np.stack(diff_crops_list) if diff_crops_list else np.zeros((0, h, w), dtype=np.float32)
    return graph_dataset, diff_crops


def _build_position_video_graphs(
    graph_builder,
    df: pd.DataFrame,
    crops_arr: np.ndarray,
    target_column: Optional[str],
    split_tracks: bool,
) -> Tuple[List[Data], np.ndarray]:
    """Build position-node graphs with one crop index per node.

    Each node is a detection at frame t with position (x, y). ``crop_idx``
    directly indexes into ``crops_arr`` (1-1 correspondence).

    Returns
    -------
    graph_dataset : list[Data]
    crops_arr : np.ndarray — the original crops array (unchanged)
    """
    from trajan.graph import get_subgraphs

    lookup = dict(zip(
        zip(df["set"], df["label"], df["frame"]),
        df["crop_idx"],
    ))

    if target_column is not None:
        df = df.copy()
        df["target"] = pd.Categorical(df[target_column]).codes

    graph_dataset: List[Data] = []

    for current_video in df["set"].unique():
        df_video = df[df["set"] == current_video].sort_values("frame").reset_index(drop=True)

        positions  = df_video[["x", "y"]].to_numpy(dtype=np.float32)
        node_labels = df_video["label"].to_numpy()
        frames     = df_video["frame"].to_numpy()

        crop_indices = np.array([
            lookup.get((current_video, int(node_labels[i]), int(frames[i])), -1)
            for i in range(len(positions))
        ], dtype=np.int64)

        if split_tracks:
            edge_index, edge_attr = graph_builder.get_connectivity(positions, frames, node_labels)
        else:
            edge_index, edge_attr = graph_builder.get_connectivity(positions, frames)

        edge_gt = graph_builder.get_gt_connectivity(node_labels, edge_index, frames)

        graph_label = int(df_video["target"].iloc[0]) if target_column is not None else []

        n_edges = len(edge_index)
        edge_index_t = (
            torch.tensor(edge_index.T, dtype=torch.long)
            if n_edges > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )
        y = (
            torch.tensor(edge_gt[:, None], dtype=torch.float)
            if n_edges > 0
            else torch.zeros((0, 1), dtype=torch.float)
        )
        edge_attr_t = (
            torch.tensor(edge_attr, dtype=torch.float)
            if n_edges > 0
            else torch.zeros((0, 3), dtype=torch.float)
        )

        graph = Data(
            x=torch.tensor(positions, dtype=torch.float),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            frames=torch.tensor(frames, dtype=torch.float),
            y=y,
            graph_label=torch.tensor(graph_label, dtype=torch.int64),
            crop_idx=torch.tensor(crop_indices, dtype=torch.long),
        )

        if split_tracks:
            graph_dataset += get_subgraphs(graph)
        else:
            graph_dataset.append(graph)

    return graph_dataset


def build_video_graph_dataset(
    graph_builder,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    split_tracks: bool = True,
) -> List[Data]:
    """Build a graph dataset with per-node crop indices for video-fused training.

    Dispatches to velocity or position mode based on the type of
    ``graph_builder``. In velocity mode each node carries ``crop_idx``
    (one index). In position mode each node carries both ``crop_idx`` and
    ``crop_idx_next`` so that ``VideoGraphDataset`` can form frame-difference
    crops.

    Parameters
    ----------
    graph_builder : VelocityGraphFromTrajectories or PositionGraphFromTrajectories
    df : pd.DataFrame
        DataFrame with columns ``[set, label, frame, crop_idx, ...]``.
    target_column : str, optional
    split_tracks : bool, optional

    Returns
    -------
    list[Data]
    """
    from trajan.graph import VelocityGraphFromTrajectories

    if isinstance(graph_builder, VelocityGraphFromTrajectories):
        return _build_velocity_video_graphs(graph_builder, df, target_column, split_tracks)
    return _build_position_video_graphs(graph_builder, df, target_column, split_tracks)


class VideoGraphDataset(_GraphDatasetBase):
    """Graph dataset pairing each subgraph with per-node image crops.

    Inherits all sampling infrastructure from ``_GraphDatasetBase`` and
    overrides ``__getitem__`` to also return a crop tensor.

    Supports both velocity-node and position-node graphs:

    - **Velocity mode** (``node_features="velocity"``): each node's visual
      feature is the crop centred on the particle at the velocity step's end
      frame. Node features are standardised by ``velocity_std``.

    - **Position mode** (``node_features="position"``): each node's visual
      feature is the frame-difference crop
      ``crops[crop_idx_next] − crops[crop_idx]``, shifted to [0, 1] via
      ``(diff + 1) / 2``. Node features follow ``PositionGraphDataset``
      normalisation (per-subgraph step std + centring).

    Parameters
    ----------
    graph_dataset : list[Data]
        Output of ``build_video_graph_dataset``.
    crops : np.ndarray or torch.Tensor
        Pre-normalised crop array of shape ``(N, H, W)``. Crops are assumed
        to be already in an appropriate range (e.g. [0, 1]).
    Dt_range : tuple[int, int]
        Min/max temporal window length.
    dataset_size : int
        Samples per epoch.
    velocity_std : float
        Used only in velocity mode for node-feature standardisation.
    transform : callable, optional
    target : {"edges", "nodes", "global"}
    sample_balanced : bool
    node_features : {"velocity", "position"}
        Selects crop mode and normalisation strategy.
    crop_size : int, optional
        If given, crops are resized to ``(crop_size, crop_size)`` using
        bilinear interpolation.
    frame_rate : float, optional
        Stored as metadata; does not affect processing.
    """

    def __init__(
        self,
        graph_dataset: list,
        crops,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        velocity_std: float = 1.0,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = False,
        node_features: Literal["velocity", "position"] = "velocity",
        crop_size: Optional[int] = None,
        frame_rate: Optional[float] = None,
    ) -> None:
        super().__init__(graph_dataset, Dt_range, dataset_size, transform, target, sample_balanced)
        self.velocity_std = velocity_std
        self.node_features = node_features
        self.frame_rate = frame_rate

        # Convert to (N, 1, H, W) tensor and optionally resize; no normalisation here
        crops_t = torch.tensor(np.asarray(crops, dtype=np.float32))
        if crops_t.ndim == 3:
            crops_t = crops_t.unsqueeze(1)
        if crop_size is not None and crops_t.shape[-1] != crop_size:
            crops_t = F.interpolate(
                crops_t, size=(crop_size, crop_size),
                mode="bilinear", align_corners=False,
            )
        self.crops = crops_t

    # _process_subgraph satisfies the base-class contract but is never called
    # because __getitem__ is overridden below.
    def _process_subgraph(self, graph: Data) -> Data:
        raise NotImplementedError

    def __getitem__(self, _: int) -> Tuple[Data, torch.Tensor]:
        graph = self._sample_subgraph()  # crop_idx (and crop_idx_next) forwarded automatically

        imgs = self.crops[graph.crop_idx]  # (N, 1, H, W)

        if self.node_features == "velocity":
            imgs_next = self.crops[graph.crop_idx_next]  # (N, 1, H, W)
            all_crops = torch.cat([imgs, imgs_next])      # (2N, 1, H, W) — normalise jointly
        else:
            all_crops = imgs

        # Normalise all crops in the subgraph together (2nd–98th percentile)
        lo, hi = torch.quantile(all_crops.flatten(), torch.tensor([0.02, 0.98]))
        scale = (hi - lo).clamp(min=1e-6)
        all_crops = ((all_crops - lo) / scale).clamp(0.0, 1.0)

        if self.node_features == "velocity":
            n = len(imgs)
            imgs = (all_crops[n:] - all_crops[:n] + 1.0) / 2.0  # shift [-1, 1] → [0, 1]
        else:
            imgs = all_crops

        if self.transform:
            graph, imgs = self.transform(graph, imgs)

        graph = self._compute_graph_features(graph)
        graph = self._normalize(graph)

        return graph, imgs

    def _compute_graph_features(self, graph: Data) -> Data:
        if self.node_features == "velocity":
            velocities = graph.x
            speeds = torch.norm(velocities, dim=1)
            if velocities.shape[0] > 1:
                v_norm = velocities / (speeds.unsqueeze(1) + 1e-10)
                persistence = (v_norm[:-1] * v_norm[1:]).sum(dim=1).mean()
            else:
                persistence = torch.zeros(1, device=velocities.device).squeeze()
            graph.persistence = persistence.unsqueeze(0)
            graph.graph_features = persistence.unsqueeze(0).unsqueeze(0)
        else:
            coords = graph.x
            steps = torch.diff(coords, dim=0)
            step_sizes = torch.norm(steps, dim=1)
            local_scale = step_sizes.mean() + 1e-10
            T = coords.shape[0]
            centered_raw = coords - coords.mean(dim=0)
            rg = torch.norm(centered_raw, dim=1).pow(2).mean().sqrt()
            expected_rg = torch.sqrt(local_scale ** 2 * T / 2)
            normalized_rg = rg / (expected_rg + 1e-10)
            graph.local_scale = local_scale.unsqueeze(0)
            graph.normalized_rg = normalized_rg.unsqueeze(0)
            graph.graph_features = torch.stack([local_scale, normalized_rg]).unsqueeze(0)
        return graph

    def _normalize(self, graph: Data) -> Data:
        graph = graph.clone()
        if self.node_features == "velocity":
            graph.x = graph.x / self.velocity_std  # no centering: velocities are location-invariant
        else:
            step_sizes = torch.norm(torch.diff(graph.x, dim=0), dim=1)
            local_scale = step_sizes.mean() + 1e-10
            step_std = step_sizes.std() if step_sizes.numel() > 1 else local_scale - 1e-10
            step_std = step_std + 1e-10
            graph.x = graph.x / step_std
            graph.x = graph.x - graph.x.mean(dim=0)
        return graph
