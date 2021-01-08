import torch
import copy
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import List

@dataclass
class Tracker:
    """This class tracks all the operations of a given module by performing a forward pass. 
    
    Example:

        >>> import torch
        >>> import torch.nn as nn
        >>> from glasses.utils import Tracker
        >>> model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
        >>> tr = Tracker(model)
        >>> tr(x)
        >>> print(tr.traced) # all operations
        >>> print('-----')
        >>> print(tr.parametrized) # all operations with learnable params
        
        outputs 
        
        ``[Linear(in_features=1, out_features=64, bias=True),
        ReLU(),
        Linear(in_features=64, out_features=10, bias=True),
        ReLU()]
        -----
        [Linear(in_features=1, out_features=64, bias=True),
        Linear(in_features=64, out_features=10, bias=True)]``
    """
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = len(list(m.modules())) == 1 or  isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor) -> Tensor:
        for m in self.module.modules():
            self.handles.append(
                m.register_forward_hook(self._forward_hook))
        self.module(x)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))
