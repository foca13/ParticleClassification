import numpy as np
import torch_geometric
import torch
from math import pi, sin, cos


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
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other
        attributes.
        
    Methods
    -------
    `__call__(graph)`
        Performs the random rotation on the input graph.
    
    """

    def __init__(self, p: float = 1):
        self.p = p

    def __call__(
        self: "RandomRotation",
        graph: torch_geometric.data.Data,
    ) -> "torch_geometric.data.Data":
        """Perform the random rotation.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
            
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with rotated node features.
        
        """

        if np.random.random() > self.p:
            return graph
        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons
        angle = np.random.rand() * 2 * pi
        rotation_matrix = torch.tensor(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        ).float()
        rotated_node_attr = torch.matmul(node_feats, rotation_matrix)
        graph.x[:, :2] = rotated_node_attr + 0.5  # Restored positons
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

    Methods
    -------
    `__call__(graph)`
        Performs the random flip on the input graph.
    
    """

    def __init__(self, p: float = 1):
        self.p = p

    def __call__(
        self: "RandomFlip",
        graph: torch_geometric.data.Data,
    ) -> "torch_geometric.data.Data":
        """Perform the random flip.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
        
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with flipped node features.

        """
        if np.random.random() > self.p:
           return graph 
        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons
        if np.random.randint(2): node_feats[:, 0] *= -1
        if np.random.randint(2): node_feats[:, 1] *= -1
        graph.x[:, :2] = node_feats + 0.5  # Restored positons
        return graph


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
        graph.distance = graph.distance[~edges_connected_to_removed_node]
        graph.y = graph.y[~edges_connected_to_removed_node]

        return graph
