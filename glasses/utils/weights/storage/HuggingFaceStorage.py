from pathlib import Path
from dataclasses import dataclass
from typing import List
from torch import nn
from .hubs import HFModelHub
from .Storage import Storage
from glasses.types import StateDict


@dataclass
class HuggingFaceStorage(Storage):

    ORGANIZATION: str = "glasses"
    root: Path = Path("/tmp/")

    def put(self, key: str, model: nn.Module):
        HFModelHub.save_pretrained(
            model,
            config={},
            save_directory=self.root / key,
            model_id=key,
            push_to_hub=True,
            organization=self.ORGANIZATION,
        )

    def get(self, key: str) -> StateDict:
        state_dict = HFModelHub.from_pretrained(f"{self.ORGANIZATION}/{key}")
        return state_dict

    @property
    def models(self) -> List[str]:
        return [
            "dummy",
            "resnet18",
            "resnet26",
            "resnet26d",
            "resnet34",
            "resnet34d",
            "resnet50",
            "resnet50d",
            "resnet101",
            "resnet152",
            "cse_resnet50",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
            "eca_resnet26t",
            "eca_resnet50t",
            "eca_resnet50d",
            "eca_resnet101d",
            "regnetx_002",
            "regnetx_004",
            "regnetx_006",
            "regnetx_008",
            "regnetx_016",
            "regnetx_032",
            "regnetx_040",
            "regnetx_064",
            "regnety_002",
            "regnety_004",
            "regnety_006",
            "regnety_008",
            "regnety_016",
            "regnety_032",
            "regnety_040",
            "regnety_064",
            "densenet121",
            "densenet169",
            "densenet201",
            "densenet161",
            "vgg11",
            "vgg13",
            "vgg16",
            "vgg19",
            "vgg11_bn",
            "vgg13_bn",
            "vgg16_bn",
            "vgg19_bn",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_lite0",
            "vit_base_patch16_224",
            "vit_base_patch16_384",
            "vit_base_patch32_384",
            "vit_huge_patch16_224",
            "vit_huge_patch32_384",
            "vit_large_patch16_224",
            "vit_large_patch16_384",
            "vit_large_patch32_384",
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
            "deit_base_patch16_384",
        ]
