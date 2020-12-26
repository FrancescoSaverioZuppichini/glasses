from abc import abstractclassmethod
from functools import partial
from typing import Callable

from torch import Tensor, nn

from ...base import VisionModule


class SegmentationModule(VisionModule):
    """Base Segmentation Module class
    """

    def __init__(self, in_channels: int, n_classes: int,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 **kwargs):

        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.decoder = decoder(lateral_widths=self.encoder.features_widths[::-1],
                               start_features=self.encoder.widths[-1],
                               **kwargs)
        self.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # first features call activates the storage
        self.encoder.features
        x = self.encoder(x)
        # encoder must have a .features
        features = self.encoder.features
        self.residuals = features[::-1]
        self.residuals.extend(
            [None] * (len(self.decoder.layers) - len(self.residuals)))

        x = self.decoder(x, self.residuals)

        x = self.head(x)
        return x

    @abstractclassmethod
    def from_encoder(cls, model: Callable, *args, **kwargs) -> nn.Module:
        """Extract the decoder part from a given model.

        Args:
            name (str): A function returning a model

        Returns:
            [nn.Module] A PyTorch module: 
        """
        def extract_encoder(*args, **kwargs):
            try:
                encoder = model(*args, **kwargs).encoder
            except AttributeError:
                raise AttributeError(f'Field .encoder was not found for {model}. Are you using a model from glasses.models?')
            return encoder

        return cls( *args, encoder=extract_encoder, **kwargs)
