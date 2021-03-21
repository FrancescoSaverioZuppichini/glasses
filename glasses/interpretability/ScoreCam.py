import torch
import torch.nn.functional as F
from torch import nn
from .Interpretability import Interpretability
from .GradCam import GradCamResult
from glasses.utils.Storage import ForwardModuleStorage
from .utils import find_last_layer
from typing import Callable


class ScoreCam(Interpretability):
    """
    Implementation of ScoreCam proposed in `Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks <https://arxiv.org/abs/1910.01279>`_
    """

    def __call__(
        self,
        x: torch.Tensor,
        module: nn.Module,
        layer: nn.Module = None,
        target: int = None,
        postprocessing: Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> GradCamResult:
        """Run GradCam on the input given a model

        Args:
            x (torch.Tensor): Input tensor, e.g. an image
            module (nn.Module): Model
            layer (nn.Module, optional): The layer we wish to interpreter, if `None` then the last conv layer will be used. Defaults to None.
            target (int, optional): The target tensor, if `None` the model output (after softmax and argmax) wil be used. Defaults to None.
            postprocessing (Callable[[torch.Tensor], torch.Tensor], optional): A function used to post process the output, e.g. de-normalize. Defaults to None.

        Returns:
            GradCamResult: The result of the scorecam, you can call `.show` to see it.
        """
        layer = find_last_layer(x, module, nn.Conv2d) if layer is None else layer
        # register forward and backward storages
        features_storage = ForwardModuleStorage(module, [layer])
        with torch.no_grad():
            out = module(x)
        # get back the weights
        features = features_storage[layer]
        _, c, _, _ = features.shape
        if target is None:
            target = torch.argmax(torch.softmax(out, dim=1))
        acts = F.relu(features)
        # rescale the activations to match the input size
        acts_up = F.interpolate(
            acts, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        # normalize
        acts_up = (acts_up - acts_up.min()) / (acts_up.max() - acts_up.min())
        # repeat the input matching the number of acts
        batch = x.repeat(c, 1, 1, 1)
        # [1, N, H, W] -> [C, N, H, W]
        batch *= acts_up.repeat(x.shape[1], 1, 1, 1).permute(1, 0, 2, 3)

        with torch.no_grad():
            out = module(batch)
            score = out[:, target]
            features_score = F.softmax(score, dim=0)
        # weight eatch feature
        features_score = features_score.view(1, -1, 1, 1)
        cam = F.relu(torch.sum(acts * features_score, dim=1)).squeeze()
        return GradCamResult(x.detach(), cam, postprocessing)
