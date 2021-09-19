import torch
import torch.nn as nn
from glasses.utils.Storage import ForwardModuleStorage
from torchinfo import summary
from typing import Dict, Tuple, List
from .protocols import Freezable, Interpretable


class VisionModule(nn.Module, Freezable, Interpretable):
    """Base vision module, all models should subclass it."""

    def summary(self, input_shape: Tuple[int]=(1, 3, 224, 224), device: Optional[torch.device] = None):
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


class Encoder(nn.Module):
    """Base encoder class, it allows to access the inner features."""

    def __init__(self):
        super().__init__()
        self.storage = None

    @property
    def stages(self) -> List[nn.Module]:
        return self.layers[:-1]

    @property
    def features_widths(self) -> List[int]:
        return self.widths[:-1]

    @property
    def features(self) -> List[torch.Tensor]:
        if self.storage is None:
            self.storage = ForwardModuleStorage(self, [*self.stages])
        return list(self.storage.state.values())
