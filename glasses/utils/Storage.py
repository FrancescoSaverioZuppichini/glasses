import torch
import numpy as np
from functools import partial, reduce
from collections import OrderedDict
from pprint import pprint

class MutipleKeysDict(OrderedDict):
    """
    Allow to get values from multiple keys. Example:
    
    ```python
    d = MutipleKeysDict({ 'a' : 1, 'b' : 2, 'c' : 3})
    d[['a', 'b']]
    # out [1,2]
    ```
    """
    def __getitem__(self, keys):
        
        if type(keys) is list:
            item = [dict.__getitem__(self, key) for key in keys]
        else: item = super().__getitem__(keys)
        # # if is a list and contains only one el, return it
        # if type(item) is list and len(item) == 1: item = item[0]
        return item

class ModuleStorage():
    def __init__(self, where2layers, debug=False):
        self.where2layers = where2layers
        self.where = list(self.names)[0]
        self.state = self._state
        self.unsubcribe = []
        self.debug = debug
    
    @property
    def _state(self):
        return MutipleKeysDict({ 
            k : MutipleKeysDict() if type(self.where2layers) == dict else [] 
            for k in self.names 
        })
    
    @property
    def names(self):
        names = []
        if type(self.where2layers) == dict:
            names = self.where2layers.keys()
        elif type(self.where2layers) is list:
            names = self.where2layers
        return names
    
    @property
    def layers(self):
        """
        Flat all the layers in the same array
        """
        layers = []
        if type(self.where2layers) == dict:
            layers = reduce(lambda a, b: a + b, self.where2layers.values())
        elif type(self.where2layers) is list:
            layers = self.where2layers
        return layers 
    
    def register_hooks(self, how='forward'):
        """
        Loop in all the layers and register a hook. There is ONLY one hook per layer to improve
        performance.
        """
        for layer in self.layers:
            # create a hash of a layer as an identifier, this is unique
#             name = f"{type(layer).__name__.lower()}-{hash(layer)}"
            if how == 'forward':
                self.unsubcribe.append(layer.register_forward_hook(partial(self.hook, name=layer)))
            elif how == 'backward':
                self.unsubcribe.append(layer.register_backward_hook(partial(self.hook, name=layer)))
            else:
                raise ValueError("type must be 'forward' or 'backward'")
            if self.debug: print(f"[INFO] {how} hook registered to {layer}")
        
    def hook(self, m, i, o, name):
        if self.debug: print(f"{m} called")
        if type(self.where2layers) == dict:
    #       store only the outputs from the correct layers defined in self.where2layers
            if m in self.where2layers[self.where]: self.state[self.where][name] = o
        if type(self.where2layers) is list:
            self.state[name] = o
            
    def clear(self):
        if self.debug: print('[INFO] clear')
        [un.remove() for un in self.unsubcribe]

    def __call__(self, where=None, *args, **kwargs):
        if where is not None:
            if where not in self.keys(): raise KeyError(f"we cannot find any layers with key {where}")
            self.where = where
        
    def __repr__(self):
        items = lambda x: x.items() if type(x) == MutipleKeysDict else enumerate(x)
        return str({k: [{i : e.shape for i, e in items(v)}] for k, v in self.state.items()})    

    def __getitem__(self, key):
        item = self.state[key]
        return item
    
    def keys(self):
        return self.state.keys()


class ForwardModuleStorage(ModuleStorage):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module
        self.register_hooks(how='forward')
        
        
    def __call__(self, x, *args, **kwargs):
        super().__call__(*args, **kwargs)
        if type(x) != list: x = [x]
        [self.module(_x) for _x in x]
        
class BackwardModuleStorage(ModuleStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_hooks(how='backward')
        
    def __call__(self, x, *args, **kwargs):
        super().__call__(*args, **kwargs)
        if type(x) != list: x = [x]
        [_x.backward() for _x in x]