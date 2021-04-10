import torch
import os
import logging
import torch.nn as nn
from torch import nn
from dataclasses import dataclass
from typing import Dict
from torch import Tensor
from pathlib import Path
from typing import Callable
from functools import wraps
from .HFModelHub import HFModelHub


ORGANIZATION_NAME = "glasses"


StateDict = Dict[str, Tensor]


def pretrained(name: str = None) -> Callable:
    _name = name

    def decorator(func: Callable) -> Callable:
        """Decorator to fetch the pretrained model.

        Args:
            func ([Callable]): The function to which the decorator is applied

        Returns:
            [Callable]: The decorated funtion
        """
        name = func.__name__ if _name is None else _name
        provider = PretrainedWeightsProvider()

        @wraps(func)
        def wrapper(
            *args,
            pretrained: bool = False,
            excluding: Callable[[nn.Module], nn.Module] = None,
            **kwargs,
        ) -> Callable:
            model = func(*args, **kwargs)
            if pretrained:
                state_dict = provider[name]
                model = load_pretrained_model(model, state_dict, excluding)
            return model

        return wrapper

    return decorator


def load_pretrained_model(
    model: nn.Module,
    state_dict: StateDict,
    excluding: Callable[[nn.Module], nn.Module] = None,
) -> nn.Module:
    """Load the pretrained weights to the model. Optionally, you can exclude some sub-module.

    Usage:
        >>> load_pretrained_model(your_model, pretrained_state_dict)
        >>> #load the pretrained weights but not in `model.head`
        >>> load_pretrained_model(your_model, pretrained_state_dict, excluding: lambda model: model.head)

    Args:
        model (nn.Module): A PyTorch Module
        state_dict (Dict[AnyStr, Tensor]): The state dict you want to use
        excluding (Callable[[nn.Module], nn.Module], optional): [description]. A function telling which sub-module you want to exclude

    Raises:
        AttributeError: Raising if you return a wrong sub-module from `excluding`

    Returns:
        nn.Module: The model with the new state dict
    """
    excluded = None
    excluded_key = None

    if excluding is not None:
        excluded = excluding(model)

    old_state_dict = model.state_dict()
    # find the key name of the module we want to exluce
    for name, module in model.named_modules():
        if module is excluded:
            excluded_key = name

    wrong_module = excluded is not None and excluded_key is None

    if wrong_module:
        raise AttributeError(f"Model doesn't contain {excluded}")

    if excluded_key is not None:
        logging.info(f"Weights starting with `{excluded_key}` won't be loaded.")
        # copy in the new state the old weights
        for k, v in state_dict.items():
            if k.startswith(excluded_key):
                state_dict[k] = old_state_dict[k]
    # apply it to the model
    model.load_state_dict(state_dict)
    model.eval()
    return model


@dataclass
class PretrainedWeightsProvider:
    """
    This class allows to retrieve pretrained models weights (state dict).

    Example:
        >>> provider = PretrainedWeightsProvider()
        >>> provider['resnet18'] # get a pre-trained resnet18 model
        # see all the outputs
        >>> provider = PretrainedWeightsProvider(verbose=1)
        # change save dir
        >>> provider = PretrainedWeightsProvider(save_dir=Path('./awesome/'))
        # override model even if already downloaded
        >>> provider = PretrainedWeightsProvider(override=True)
    """

    BASE_DIR: Path = Path(torch.hub.get_dir()) / Path("glasses")
    save_dir: Path = BASE_DIR
    verbose: int = 0
    override: bool = False

    def __post_init__(self):
        try:
            self.save_dir.mkdir(exist_ok=True)
        except FileNotFoundError:
            default_dir = str(Path(__file__).resolve().parent)
            self.save_dir = Path(os.environ.get("HOME", default_dir)) / Path(
                ".glasses/"
            )
            self.save_dir.mkdir(exist_ok=True)
        os.environ["GLASSES_HOME"] = str(self.save_dir)
        # we also need to know which models are pretrained
        with open("pretrained_models.txt", "r") as f:
            data = f.read()
            self.weights_zoo = data.split(",")

    def __getitem__(self, key: str) -> StateDict:
        weights = HFModelHub.from_pretrained(f"{ORGANIZATION_NAME}/{key}")
        return weights
