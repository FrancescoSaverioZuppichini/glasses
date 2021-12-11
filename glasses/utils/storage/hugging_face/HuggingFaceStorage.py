from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from requests.models import Response
from torch import nn
from .HFModelHub import HFModelHub
from huggingface_hub import HfApi, Repository
from ..Storage import Storage
from glasses.types import StateDict
from glasses.logger import logger
import requests


@dataclass
class HuggingFaceStorage(Storage):

    ORGANIZATION: str = "glasses"
    root: Path = Path("/tmp/hf_bub/")

    def __post_init__(self):
        self._models: Optional[List[str]] = None
        self._api: HfApi = HfApi()

    def get_models(self) -> List[str]:
        endpoint: str = self._api.endpoint
        res: Response = requests.get(
            f"{endpoint}/api/models", params={"author": self.ORGANIZATION}
        )
        res.raise_for_status()
        models: List[Dict] = res.json()

        # modelId has the following form: <ORGANIZATION>/<REPO_NAME>/<FILE_NAME>
        names: List[str] = [e["modelId"].split("/")[1] for e in models]

        return names

    def put(self, key: str, model: nn.Module):
        save_directory: str = self.root / key
        repo_url = None
        if key in self.models:
            # if a model is already stored, attempting to push a new one will
            # trigger and error since a git repo already exists. We need to first
            # clone the existing repo, and then change the files inside it
            logger.info(f"Model {key} already stored, updating the repo.")
            repo_url = f"{self._api.endpoint}/{self.ORGANIZATION}/{key}"
            # clone the repo
            Repository(save_directory, clone_from=repo_url)

        HFModelHub.save_pretrained(
            model,
            config={},
            save_directory=self.root / key,
            model_id=key,
            push_to_hub=True,
            organization=self.ORGANIZATION,
            repo_url=repo_url,
        )

    def get(self, key: str) -> StateDict:
        state_dict: StateDict = HFModelHub.from_pretrained(f"{self.ORGANIZATION}/{key}")
        return state_dict

    @property
    def models(self) -> List[str]:
        if self._models is None:
            self._models = self.get_models()
        models: List[str] = self._models
        return models
