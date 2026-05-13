from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


class TargetMode(Enum):
    EDGES = "edges"
    NODES = "nodes"
    GLOBAL = "global"


def _load_crops(crops) -> Optional[torch.Tensor]:
    """Convert a crop array to a (N, 1, H, W) float32 tensor, or return None."""
    if crops is None:
        return None
    crops_t = torch.tensor(np.asarray(crops, dtype=np.float32))
    if crops_t.ndim == 3:
        crops_t = crops_t.unsqueeze(1)
    return crops_t


def _normalize_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """2nd–98th percentile normalisation of a crop batch to [0, 1]."""
    lo, hi = torch.quantile(imgs.flatten(), torch.tensor([0.02, 0.98]))
    return ((imgs - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Shared sampling base class
# ---------------------------------------------------------------------------

class _GraphDatasetBase(torch.utils.data.Dataset):
    """Shared temporal-sampling infrastructure for trajectory graph datasets.

    Handles graph selection, time-window extraction, edge/node masking, and
    optional per-node crop retrieval.

    Subclasses implement ``_process_subgraph`` to apply mode-specific feature
    computation and normalisation. They must NOT call ``self.transform`` there;
    the transform is applied here in ``__getitem__`` before ``_process_subgraph``
    is called.

    Parameters
    ----------
    graph_dataset : list[Data]
    Dt_range : tuple[int, int]
    dataset_size : int
    transform : callable, optional
        When ``crops`` is None: called as ``transform(graph)`` → ``graph``.
        When ``crops`` is provided: called as ``transform(graph, imgs)``
        → ``(graph, imgs)``.
    target : {"edges", "nodes", "global"}
    sample_balanced : bool
    crops : array-like, optional
        Crop array of shape ``(N, H, W)``. When provided, ``__getitem__``
        returns ``(graph, imgs)``; otherwise returns ``graph``.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
        crops=None,
    ) -> None:
        self.Dt_range = Dt_range
        self.graph_dataset = [g for g in graph_dataset if len(g.x) >= Dt_range[0]]
        self.dataset_size = dataset_size
        self.transform = transform
        self._target = None
        self.target = target
        self.labels = np.array([g.graph_label.item() for g in self.graph_dataset])
        self.sample_balanced = sample_balanced
        self.crops = _load_crops(crops)

    @property
    def target(self) -> TargetMode:
        return self._target

    @target.setter
    def target(self, value) -> None:
        self._target = TargetMode(value) if isinstance(value, str) else value

    def __len__(self) -> int:
        return self.dataset_size

    def _sample_subgraph(self) -> Data:
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

        # Forward all node-level attributes (including crop_idx)
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

    def __getitem__(self, _: int):
        graph = self._sample_subgraph()

        if self.crops is not None:
            imgs = self.crops[graph.crop_idx]  # (N, 1, H, W)
            if self.transform:
                graph, imgs = self.transform(graph, imgs)
            graph = self._process_subgraph(graph)
            imgs = _normalize_imgs(imgs)
            return graph, imgs

        if self.transform:
            graph = self.transform(graph)
        graph = self._process_subgraph(graph)
        return graph


# ---------------------------------------------------------------------------
# Velocity-node dataset
# ---------------------------------------------------------------------------

class VelocityGraphDataset(_GraphDatasetBase):
    """Graph dataset for velocity-node graphs (VelocityGraphFromTrajectories).

    Parameters
    ----------
    graph_dataset : list[Data]
    Dt_range : tuple[int, int]
    dataset_size : int
    velocity_std : float
    transform : callable, optional
    target : {"edges", "nodes", "global"}
    sample_balanced : bool
    crops : array-like, optional
        When provided, ``__getitem__`` returns ``(graph, imgs)``.
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
        crops=None,
    ) -> None:
        super().__init__(graph_dataset, Dt_range, dataset_size, transform, target, sample_balanced, crops)
        self.velocity_std = velocity_std

    def _process_subgraph(self, graph: Data) -> Data:
        velocities = graph.x
        speeds = torch.norm(velocities, dim=1)
        if velocities.shape[0] > 1:
            v_norm = velocities / (speeds.unsqueeze(1) + 1e-10)
            persistence = (v_norm[:-1] * v_norm[1:]).sum(dim=1).mean()
        else:
            persistence = torch.zeros(1, device=velocities.device).squeeze()

        graph.persistence = persistence.unsqueeze(0)
        graph.graph_features = persistence.unsqueeze(0).unsqueeze(0)  # (1, 1)

        graph = graph.clone()
        graph.x = graph.x / self.velocity_std
        return graph


# ---------------------------------------------------------------------------
# Position-node dataset
# ---------------------------------------------------------------------------

class PositionGraphDataset(_GraphDatasetBase):
    """Graph dataset for position-node graphs (PositionGraphFromTrajectories).

    Parameters
    ----------
    graph_dataset : list[Data]
    Dt_range : tuple[int, int]
    dataset_size : int
    transform : callable, optional
    target : {"edges", "nodes", "global"}
    sample_balanced : bool
    crops : array-like, optional
        When provided, ``__getitem__`` returns ``(graph, imgs)``.
    """

    def __init__(
        self,
        graph_dataset: list,
        Dt_range: Tuple[int, int],
        dataset_size: int,
        transform: callable = None,
        target: Literal["edges", "nodes", "global"] = "global",
        sample_balanced: bool = True,
        crops=None,
    ) -> None:
        super().__init__(graph_dataset, Dt_range, dataset_size, transform, target, sample_balanced, crops)

    def _process_subgraph(self, graph: Data) -> Data:
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

        step_std = step_sizes.std() if step_sizes.numel() > 1 else local_scale - 1e-10
        step_std = step_std + 1e-10
        graph = graph.clone()
        graph.x = graph.x / step_std
        graph.x = graph.x - graph.x.mean(dim=0)
        return graph
