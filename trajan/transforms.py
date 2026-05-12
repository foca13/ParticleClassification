from __future__ import annotations

import numpy as np
import torch_geometric
import torch
from math import pi, sin, cos
from typing import Optional, Tuple, Union


class RandomRotation:
    """Random rotation to augment training data.

    This class applies a random rotation to the node features of a graph to
    augment the training data. The rotation is performed in the 2D plane, and
    the angle of rotation is randomly sampled from a uniform distribution. The
    rotation is applied to the x and y coordinates of the node features, which
    are assumed to be in the first two columns of the node feature matrix. The
    rotation is performed in place, and the modified graph is returned. The
    rotation is centered around the origin (0, 0) and the node features are
    restored to their original positions after the rotation.

    When ``imgs`` (shape ``(N, 1, H, W)``) is supplied the same rotation angle
    is applied to each crop so that velocity directions and image orientation
    stay consistent.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other
        attributes.

    Methods
    -------
    `__call__(graph, imgs=None)`
        Performs the random rotation on the input graph (and crops if given).

    """

    def __init__(self, p: float = 1):
        self.p = p

    def __call__(
        self,
        graph: torch_geometric.data.Data,
        imgs: Optional[torch.Tensor] = None,
    ) -> Union[torch_geometric.data.Data, Tuple[torch_geometric.data.Data, torch.Tensor]]:
        if np.random.random() > self.p:
            return (graph, imgs) if imgs is not None else graph

        graph = graph.clone()
        angle = np.random.rand() * 2 * pi
        rotation_matrix = torch.tensor(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        ).float()
        graph.x[:, :2] = torch.matmul(graph.x[:, :2], rotation_matrix)

        if imgs is not None:
            angle_deg = angle * 180.0 / pi
            imgs = torch.stack([
                _rotate_crop(img, angle_deg) for img in imgs
            ])
            return graph, imgs

        return graph


class RandomFlip:
    """Random flip to augment training data.

    This class applies a random flip to the node features of a graph to
    augment the training data. The flip is performed in the 2D plane, and the
    flip is applied to the x and y coordinates of the node features, which are
    assumed to be in the first two columns of the node feature matrix. The
    flip is performed in place, and the modified graph is returned. The flip
    is centered around the origin (0, 0) and the node features are restored to
    their original positions after the flip.

    When ``imgs`` (shape ``(N, 1, H, W)``) is supplied the same flip axes are
    applied to each crop so that velocity directions and image orientation stay
    consistent.

    Methods
    -------
    `__call__(graph, imgs=None)`
        Performs the random flip on the input graph (and crops if given).

    """

    def __init__(self, p: float = 1):
        self.p = p

    def __call__(
        self,
        graph: torch_geometric.data.Data,
        imgs: Optional[torch.Tensor] = None,
    ) -> Union[torch_geometric.data.Data, Tuple[torch_geometric.data.Data, torch.Tensor]]:
        if np.random.random() > self.p:
            return (graph, imgs) if imgs is not None else graph

        graph = graph.clone()
        flip_x = bool(np.random.randint(2))
        flip_y = bool(np.random.randint(2))

        if flip_x:
            graph.x[:, 0] *= -1
            if imgs is not None:
                imgs = torch.flip(imgs, dims=[3])  # horizontal flip (W axis)
        if flip_y:
            graph.x[:, 1] *= -1
            if imgs is not None:
                imgs = torch.flip(imgs, dims=[2])  # vertical flip (H axis)

        return (graph, imgs) if imgs is not None else graph


def _rotate_crop(img: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate a single (1, H, W) crop by angle_deg degrees (counterclockwise)."""
    # Build affine grid for rotation around the centre.
    angle_rad = torch.tensor(angle_deg * pi / 180.0)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    # Affine matrix for counterclockwise rotation (no translation).
    theta = torch.tensor([[cos_a, -sin_a, 0.0],
                          [sin_a,  cos_a, 0.0]], dtype=torch.float32)
    grid = torch.nn.functional.affine_grid(
        theta.unsqueeze(0), img.unsqueeze(0).size(), align_corners=False
    )
    rotated = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return rotated.squeeze(0)


class NodeDropout:
    """Removal (dropout) of random nodes to simulate missing frames.

    This class randomly removes nodes from a graph to simulate missing frames.
    The dropout is performed by randomly selecting a subset of nodes to remove
    based on a specified dropout rate. The edges, weights, labels, and
    distances connected to the removed nodes are also removed. The modified
    graph is returned with the remaining nodes and edges. The dropout is
    performed in place, and the original graph is unchanged. The dropout rate
    is specified as a parameter, and the random selection of nodes to remove is
    performed using a uniform distribution. The removed nodes are not restored
    to their original positions, and the modified graph is returned with the
    remaining nodes and edges.
  
    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other
        attributes.
  
    Methods
    -------
    `__call__(graph)`
        Performs the node dropout on the input graph.

    """

    def __init__(self, p: float = 1):
        self.p = p

    def __call__(
        self: "NodeDropout", 
        graph: torch_geometric.data.Data,
    ) -> "torch_geometric.data.Data":
        """Perform the node dropout.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
    
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with the remaining nodes and edges after
            the dropout.

        """

        # Ensure original graph is unchanged.
        graph = graph.clone()

        # Get indices of random nodes.
        idx = np.array(list(range(len(graph.x))))
        dropped_idx = idx[np.random.rand(len(graph.x)) < self.p]

        # Compute connectivity matrix to dropped nodes.
        for dropped_node in dropped_idx:
            edges_connected_to_removed_node = np.any(
                np.array(graph.edge_index) == dropped_node, axis=0
            )

        # Remove edges, weights, labels connected to dropped nodes with the
        # bitwise not operator '~'.
        graph.edge_index = \
            graph.edge_index[:, ~edges_connected_to_removed_node]
        graph.edge_attr = graph.edge_attr[~edges_connected_to_removed_node]
        graph.y = graph.y[~edges_connected_to_removed_node]

        return graph
