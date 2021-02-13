import math

from typing import List, Union

class CompoundScaler:
    """Scales `widths` and `depths`. Proposed in `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
 <https://arxiv.org/abs/1905.11946>`_.

    Examples:
        >>> scaler = CompoundScaler()
        >>> scaler(1.1, 1.2, [32, 64, 128], [2,2,3])

    """

    def make_divisible(self, value: float, divisor:int=8) -> int:
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def width_scaling(self, width: int, width_multi: float) -> int:
        scaled = width if width == 1 else int(
            self.make_divisible(width * width_multi))
        return scaled

    def depth_scaling(self, depth: int, depth_multi: float) -> int:
        scaled = int(math.ceil(depth * depth_multi))
        return scaled

    def __call__(self, width_factor: float, depth_factor: float, widths: List[int], depths: List[int]) -> Union[List[int], List[int]]:
        """Scale up `widhts` and `depths` using `width_factor` and `depth_factor`

        Args:
            width_factor (float): value used to scale up widths
            depth_factor (float): value used to scale up depths
            widths (List[int]): array where each value is the number of in_features in one layer
            depths (List[int]): array where each value is the number of blocks in one layer

        Returns:
            Union[List[int], List[int]]: scaled widths and depths
        """

        widths = [self.width_scaling(w, width_factor) for w in widths]
        depths = [self.depth_scaling(d, depth_factor) for d in depths]
        

        return widths, depths