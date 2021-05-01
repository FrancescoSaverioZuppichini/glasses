from typing import Any


class Storage:
    def put(self, *args: Any, **kwargs: Any):
        raise NotImplemented

    def get(self, key: str, **kwargs: Any) -> Any:
        raise NotImplemented

    def __contains__(self, key: str) -> bool:
        raise NotImplemented
