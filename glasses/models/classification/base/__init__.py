from torch import Tensor, nn

from ...base import VisionModule


class ClassificationModule(VisionModule):
    """Base Classification Module class"""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        in_channels: int = 3,
        n_classes: int = 1000,
        **kwargs
    ):

        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.head = head(self.encoder.widths[-1], n_classes)
        self.initialize()

    def initialize(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x
