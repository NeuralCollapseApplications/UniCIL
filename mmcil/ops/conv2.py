from typing import Union, Tuple, TypeVar

import torch
import torch.nn as nn

from mmcv.cnn import Conv2d, CONV_LAYERS

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


@CONV_LAYERS.register_module('Conv2', force=False)
class Conv2dx2(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            **kwargs
    ):
        super().__init__()
        self.conv_a = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )
        self.conv_b = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs
        )
        self.alpha = nn.Parameter(torch.tensor(0., dtype=torch.float32))

    def forward(self, x):
        return self.conv_a(x) * self.alpha.sigmoid() + self.conv_b(x) * (1 - self.alpha.sigmoid())
