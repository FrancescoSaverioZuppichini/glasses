import torch
import torch.nn as nn
import inspect
from pathlib import Path
from torchsummary import summary
from glasses.utils.PretrainedWeightsProvider import Config
from .protocols import Freezable, Interpretable
from typing import Dict
from glasses.utils.Storage import ForwardModuleStorage


class VisionModule(nn.Module, Freezable, Interpretable):
    """Base vision module, all models should subclass it.
    """
    configs: Dict[str, Config] = {}

    def summary(self, input_shape=(3, 224, 224), device: torch.device = None):
        """Useful method to run `torchsummary` directly from the model

        Args:
            input_shape (tuple, optional): [description]. Defaults to (3, 224, 224).

        Returns:
            [type]: [description]
        """
        device = torch.device('cuda' if torch.cuda.is_available(
        ) else 'cpu') if device is None else device
        return summary(self.to(device), input_shape, device=device)


class Encoder(nn.Module):
    """Base encoder class, it allows to access the inner features.
    """

    def __init__(self):
        super().__init__()
        self.storage = None

    @property
    def stages(self):
        return self.layers[:-1]

    @property
    def features_widths(self):
        return self.widths[:-1]

    @property
    def features(self):
        if self.storage is None:
            self.storage = ForwardModuleStorage(self, [*self.stages])
        return list(self.storage.state.values())
