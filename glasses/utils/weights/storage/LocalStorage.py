import torch
from .Storage import Storage
from dataclasses import dataclass
from glasses.types import StateDict
from pathlib import Path
from torch import nn


@dataclass
class LocalStorage(Storage):
    root: Path = Path("/tmp/glasses")
    override: bool = False

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)
        self.models_files = list(self.root.glob("*.pth"))

    def put(self, key: str, model: nn.Module):
        save_path = self.root / Path(f"{key}.pth")
        if key not in self or self.override:
            torch.save(model.state_dict(), save_path)
            assert save_path.exists()

    def get(self, key: str) -> StateDict:
        save_path = self.root / Path(f"{key}.pth")
        state_dict = torch.load(save_path)
        return state_dict

    def __contains__(self, key: str) -> bool:
        return key in [file.stem for file in self.models_files]
