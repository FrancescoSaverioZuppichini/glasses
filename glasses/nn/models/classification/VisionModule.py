import torch
import torch.nn as nn
import inspect
from pathlib import Path
from torchsummary import summary
from ....utils.PretrainedWeightsProvider import PretrainedWeightsProvider

class VisionModule(nn.Module):
    def summary(self, input_shape=(3, 224, 224)):
        """Useful method to run `torchsummary` directly from the model

        Args:
            input_shape (tuple, optional): [description]. Defaults to (3, 224, 224).

        Returns:
            [type]: [description]
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return summary(self.to(device), input_shape)

    @classmethod
    def from_pretrained(cls, name: str, save_dir: Path = PretrainedWeightsProvider.BASE_DIR) -> nn.Module:
        """This method returns a pretrained model using `name` as an identifier. 

        It use the class field `pretrained_keys` to know if a pretrained model is available with the given name.
        If yes, we first create the model then we use `PretrainedWeightsProvider` to get the correct weights and 
        finally we load them into the model.

        Args:
            name (str): [description]
            save_dir (Path, optional): [description]. Defaults to PretrainedWeightsProvider.BASE_DIR.

        Raises:
            KeyError: Raise `if a model with the given name is not available

        Returns:
            nn.Module: The pretrained model
        """
        if name not in cls.pretrained_keys:
            raise KeyError(f'model "{name}"" not found. Available models are {cls.pretrained_keys}')
        
        name_to_method = { el[0]: el[1] for el in inspect.getmembers(cls, predicate=inspect.ismethod)}
        model = name_to_method[name]()
        model.load_state_dict(PretrainedWeightsProvider(save_dir=save_dir)[name])

        return model
