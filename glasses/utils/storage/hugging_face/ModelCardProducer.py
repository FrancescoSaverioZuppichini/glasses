import os
import time
from glasses.models import AutoModel
from huggingface_hub import Repository, HfFolder
from .HuggingFaceStorage import HuggingFaceStorage
from pathlib import Path
from torch import nn

# ---
# tags:
# - image-classification
# datasets:
# - imagenet
class ModelCardProducer:
    """This class create a model card (a README.md) file for each
    model stored in the hugging face hub
    """

    models = []

    def __init__(self):
        self.root = Path("/tmp/models/")
        self.root.mkdir(exist_ok=True)
        self.storage: HuggingFaceStorage = HuggingFaceStorage()

    def __call__(self):
        for key in self.storage.models:
            self.make_model_card(key)

    def make_model_card(self, key: str):
        model_root: Path = self.root / key
        try:
            repo = Repository(
                model_root, clone_from=f"https://huggingface.co/glasses/{key}.git"
            )
            repo.git_pull()
            model: nn.Module = AutoModel.from_name(key)
            doc: str = model.__doc__
            file_path: Path = model_root / "README.rst"
            file_path_md: Path = model_root / "README.md"

            with open(file_path, "w") as f:
                f.write(doc)

            os.system(f"pandoc -s -o {str(file_path_md)} {str(file_path)}")

            with open(file_path_md, "r") as f:
                text: str = f.read()
                text = text.replace(">", "")
                text = text.replace("{.sourceCode .python}", "python")

            text = f"# {key}\n" + text
            text = text.split("Args:")[0]
            # prepend the tags, datasets\
            with open("./glasses/utils/prepend.md", "r") as f:
                text = f"{f.read()}{text}"

            with open(file_path_md, "w") as f:
                f.write(text)

            file_path.unlink()
            repo.push_to_hub()
        except OSError as e:
            print(key, e)
