from torch import nn
from torch import Tensor
from ...base import VisionModule

class ClassificationModel(VisionModule):
    """Base Segmentation Module class
    """

    def __init__(self, in_channels: int, n_classes: int,
                 encoder: nn.Module,
                 head: nn.Module,
                 **kwargs):

        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.head = head(self.encoder.widths[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x

