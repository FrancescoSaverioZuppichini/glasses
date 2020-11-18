from torch import nn
from torch import Tensor
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
