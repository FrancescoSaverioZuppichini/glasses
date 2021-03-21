from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from glasses.utils.PretrainedWeightsProvider import Config
from glasses.utils.Storage import ForwardModuleStorage
from torchinfo import summary
from functools import partial
from .protocols import Freezable, Interpretable


class VisionModule(nn.Module, Freezable, Interpretable):
    """Base vision module, all models should subclass it."""

    configs: Dict[str, Config] = {}

    def summary(self, input_shape=(1, 3, 224, 224), device: torch.device = None):
        """Useful method to run `torchinfo` directly from the model

        Args:
            input_shape (tuple, optional): [description]. Defaults to (3, 224, 224).

        Returns:
            [type]: [description]
        """
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        return summary(self.to(device), input_shape, device=device)


StagesExtractor = Callable[[nn.Module], List[nn.Module]]
WidthsExtractor = Callable[[nn.Module], List[int]]

all_stages = lambda e: e.layers[:-1]
all_widths = lambda e: e.widths[:-1]


class Encoder(nn.Module):
    """Base encoder class, it allows to access the inner features.

    Example:

            Define a dummy model by subclassing encoder

            >>> class Dummy(Encoder):
            >>>    def __init__(self, *args, **kwargs):
            >>>        super().__init__(*args, **kwargs)
            >>>        self.layers = nn.Sequential(
            >>>            nn.Linear(2, 10),
            >>>            nn.Linear(10, 20),
            >>>            nn.Linear(20, 5)
            >>>        )
            >>>    def forward(self, x):
            >>>        return self.layers(x)

            Now we can use it

            >>> d = Dummy(get_stages=lambda e: [e.layers[0]])
            >>> d.features
            >>> d(torch.randn((1, 1, 2)))
            >>> d.features # will contain only the output of the first layer

    """

    def __init__(
        self,
        get_stages: StagesExtractor = all_stages,
        get_widths: WidthsExtractor = all_widths,
    ):

        super().__init__()
        self.storage = None
        self.layers = nn.ModuleList([])
        self.widhts = []
        self.get_stages = get_stages
        self.get_widths = get_widths

    @property
    def stages(self):
        return self.get_stages(self)

    @property
    def features_widths(self):
        return self.get_widths(self)

    @property
    def features(self):
        if self.storage is None:
            self.storage = ForwardModuleStorage(self, [*self.stages])
        return list(self.storage.state.values())
