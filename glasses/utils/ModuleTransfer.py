import torch
import copy
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from .Tracker import Tracker
from pprint import pprint
from typing import List

@dataclass
class ModuleTransfer:
    """This class transfers the weight from one module to another assuming 
    they have the same set of operations but they were defined in a different way.
    
    :Examples
    
        >>> import torch
        >>> import torch.nn as nn
        >>> from eyes.utils import ModuleTransfer
        >>> model_a = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
        >>> def block(in_features, out_features):
        >>>     return nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU())
        >>> model_b = nn.Sequential(block(1,64), block(64,10))
        >>> # model_a and model_b are the same thing but defined in two different ways
        >>> x = torch.ones(1, 1)
        >>> trans = ModuleTransfer(src=model_a, dest=model_b)
        >>> trans(x)

        # now module_b has the same weight of model_a
    
    """
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input.
        Under the hood we tracked all the operations in booth modules.
        :param x: [The input to the modules]
        :type x: torch.tensor
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized
        
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))


        if len(dest_traced) != len(src_traced):
            raise Exception(
                f'Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}.')

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f'Transfered from={src_m} to={dest_m}')