from typing import Any
from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def put(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def get(self, key: str, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        pass
