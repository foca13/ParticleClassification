from typing import Sequence, Type, Union

import deeplay as dl
import torch
import torch.nn as nn
from deeplay import GlobalMeanPool


class MagikMPM(dl.GraphToGlobalMPM):
    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        pool: Union[Type[nn.Module], nn.Module] = GlobalMeanPool,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(
            hidden_features,
            out_features,
            pool,
            out_activation,
        )

    def forward(self, x):
        data = x.clone()
        x = self.encoder(x)
        x = self.backbone(x)
        x = self.selector(x)
        x = self.pool(x)
        if hasattr(data, 'graph_features'):
            x = torch.cat([x, data.graph_features], dim=-1)
        x = self.head(x)
        return x


class ImageGraphConv(dl.GraphToGlobalMPM):
    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        pool: Union[Type[nn.Module], nn.Module] = GlobalMeanPool,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
    ):
        super().__init__(
            hidden_features,
            out_features,
            pool,
            out_activation,
        )
        self.convs = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # (N, 8, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (N, 8, 12, 12)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # (N, 16, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # (N, 16, 6, 6)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (N, 32, 6, 6)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # (N, 32, 1, 1)
            nn.Flatten(),                                # (N, 32)
        )
        self.cnn_proj = nn.Linear(32, hidden_features[0])

    def forward(self, data, imgs):
        # imgs: (N_nodes, 1, H, W) — one crop per node
        img_emb = self.convs(imgs)              # (N_nodes, 32)
        img_emb = self.cnn_proj(img_emb)        # (N_nodes, hidden_features[0])

        # Run the standard GNN encoder on velocity features, then fuse image embeddings.
        # This keeps the encoder input dimension consistent (2D velocities) and adds
        # appearance information at the right abstraction level before the backbone.
        x = self.encoder(data)                  # Data with x.x of shape (N, hidden_dim)
        x.x = x.x + img_emb                    # element-wise add, both (N, hidden_dim)

        x = self.backbone(x)
        x = self.selector(x)
        x = self.pool(x)
        if hasattr(data, 'graph_features'):
            x = torch.cat([x, data.graph_features], dim=-1)
        x = self.head(x)
        return x
