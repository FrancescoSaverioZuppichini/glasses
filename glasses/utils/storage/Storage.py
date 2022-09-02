from typing import Any, List
from abc import ABC, abstractmethod


class Storage(ABC):
    @abstractmethod
    def put(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get(self, key: str, **kwargs: Any) -> Any:
        pass

    @property
    @abstractmethod
    def models(self) -> List[str]:
        pass

    def __contains__(self, key: str) -> bool:
        return key in self.models
