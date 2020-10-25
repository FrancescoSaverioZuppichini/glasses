import torch
import torch.nn as nn
import inspect
from pathlib import Path
from torchsummary import summary
from glasses.utils.PretrainedWeightsProvider import Config
from .utils.protocols import Freezable

from typing import Dict

class VisionModule(nn.Module, Freezable):
    configs: Dict[str, Config] = {}
    
    def summary(self, input_shape=(3, 224, 224), device: torch.device = None):
        """Useful method to run `torchsummary` directly from the model

        Args:
            input_shape (tuple, optional): [description]. Defaults to (3, 224, 224).

        Returns:
            [type]: [description]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        return summary(self.to(device), input_shape, device=device)

