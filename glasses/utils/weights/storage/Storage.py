from re import L
from typing import Any, List
from abc import ABC, abstractmethod
from glasses.types import StateDict


class Storage(ABC):
    @abstractmethod
    def __setitem__(self, key: str, weights: StateDict):
        pass

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        pass

    @property
    @abstractmethod
    def models(self) -> List[str]:
        pass

    def __contains__(self, key: str) -> bool:
        return key in self.models
