from pathlib import Path
from dataclasses import dataclass
from torch import nn
from .hubs import HFModelHub
from .Storage import Storage
from glasses.types import StateDict


@dataclass
class HuggingFaceStorage(Storage):

    ORGANIZATION = "glasses"
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

    def __contains__(self, key: str) -> bool:
        return False