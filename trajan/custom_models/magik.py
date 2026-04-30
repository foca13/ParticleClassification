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
