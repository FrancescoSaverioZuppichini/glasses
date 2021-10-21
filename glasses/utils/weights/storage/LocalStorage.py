import torch
from .Storage import Storage
from dataclasses import dataclass
from glasses.types import StateDict
from pathlib import Path
from torch import nn
from typing import List


@dataclass
class LocalStorage(Storage):
    root: Path = Path("/tmp/glasses")
    override: bool = False
    fmt: str = "pth"

    def __post_init__(self):
        self.root.mkdir(exist_ok=True)

    def put(self, key: str, model: nn.Module):
        save_path = self.root / Path(f"{key}.{self.fmt}")
        if key not in self or self.override:
            torch.save(model.state_dict(), save_path)
            assert save_path.exists()

    def get(self, key: str) -> StateDict:
        save_path = self.root / Path(f"{key}.{self.fmt}")
        state_dict = torch.load(save_path)
        return state_dict

    @property
    def models(self) -> List[str]:
        return [file.stem for file in self.root.glob(f"*.{self.fmt}")]
