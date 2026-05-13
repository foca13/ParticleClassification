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
    """GNN classifier with a per-node CNN branch for image crops.

    The CNN branch embeds each node's image crop and fuses it with the GNN
    encoder output before the message-passing backbone.

    Parameters
    ----------
    hidden_features : Sequence[int]
        Hidden dimensions of the GNN encoder/backbone blocks.
    out_features : int
        Number of output classes.
    pool : nn.Module
        Graph pooling layer. Default: GlobalMeanPool.
    out_activation : nn.Module or None
        Output activation (e.g. Softmax). Default: None.
    cnn_channels : Sequence[int]
        Output channels for each conv block in the CNN branch.
        One block = Conv2d → ReLU → MaxPool2d(2), except the last block
        which uses AdaptiveAvgPool2d((1, 1)) instead.
        Default: (8, 16, 32).
    kernel_size : int
        Kernel size for all conv layers. Default: 3.
    fusion : {"add", "cat"}
        How to fuse the CNN embedding with the GNN encoder output.
        ``"add"``  — element-wise addition (fewer parameters).
        ``"cat"``  — concatenation followed by a learned projection back
                     to ``hidden_features[0]`` (more expressive).
        Default: ``"add"``.
    """

    def __init__(
        self,
        hidden_features: Sequence[int],
        out_features: int,
        pool: Union[Type[nn.Module], nn.Module] = GlobalMeanPool,
        out_activation: Union[Type[nn.Module], nn.Module, None] = None,
        cnn_channels: Sequence[int] = (8, 16, 32),
        kernel_size: int = 3,
        fusion: str = "add",
    ):
        super().__init__(hidden_features, out_features, pool, out_activation)

        if fusion not in ("add", "cat"):
            raise ValueError(f"fusion must be 'add' or 'cat', got {fusion!r}")
        self.fusion = fusion

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(cnn_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            if i < len(cnn_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            in_ch = out_ch
        layers.append(nn.Flatten())
        self.convs = nn.Sequential(*layers)

        self.cnn_proj = nn.Linear(cnn_channels[-1], hidden_features[0])

        if fusion == "cat":
            self.fusion_proj = nn.Linear(2 * hidden_features[0], hidden_features[0])

    def forward(self, data, imgs):
        # imgs: (N_nodes, 1, H, W) — one crop per node
        img_emb = self.cnn_proj(self.convs(imgs))   # (N_nodes, hidden_features[0])

        x = self.encoder(data)                       # Data with x.x: (N, hidden_dim)

        if self.fusion == "add":
            x.x = x.x + img_emb
        else:
            x.x = self.fusion_proj(torch.cat([x.x, img_emb], dim=-1))

        x = self.backbone(x)
        x = self.selector(x)
        x = self.pool(x)
        if hasattr(data, "graph_features"):
            x = torch.cat([x, data.graph_features], dim=-1)
        x = self.head(x)
        return x
