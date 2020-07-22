from ..resnet import ResNetBasicBlock
from ..se import SEModule

class ResNetSEBasicBlock(ResNetBasicBlock):
    def __init__(self, in_features: int, out_features: int, reduction: int =16, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.block.add_module('se', SEModule(out_features))
        
