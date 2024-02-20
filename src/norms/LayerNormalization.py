"""
Pytorch Implementation of the Layer Normalization (https://arxiv.org/pdf/1607.06450.pdf)
"""
from torch.nn import Module, Parameter
from torch import ones, zeros
from torch import Size, Tensor
from typing import List, Union

input_shape = Union[int, List[int], Tensor]

class LayerNorm(Module):
    r""" Applies Layer Normalization to the input of the layer.

    Args:
        shape(int or list or torch.Size, required): Input shape from
                                                    an expected input of size
        eps(float, optional):                       The epsilon value for
                                                    arithmetic stability.
                                                    Default: 1e-8
        gamma(bool, optional):                      Add scale parameter.
                                                    Default: True
        offset(bool, optional):                     Add bias parameter.
                                                    Default: True

    Example:
        >>> lnorm = LayerNorm(4)
        >>> x = torch.randn((5, 4))
        >>> output = lnorm(x)
    """
    __constants__ = []
    def __init__(self, shape: input_shape, eps: float = 1e-8, gamma: bool = True,
                 offset: bool = True) -> None:
        super().__init__()
        if isinstance(shape, int):
            self.shape = (shape, )
        else:
            self.shape = (shape[-1], )
        self.shape = Size(self.shape)
        self.eps = eps
        self.gamma = gamma
        self.offset = offset
        self.__init_parameters__()

    def __init_parameters__(self) -> None:
        if self.gamma:
            self.scale = Parameter(ones(self.shape))
            self.register_parameter('scale', self.scale)
        if self.offset:
            self.bias = Parameter(zeros(self.shape))
            self.register_parameter('bias', self.bias)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        norm = (x - mean)/(var + self.eps).sqrt()
        if self.gamma:
            norm *= self.scale
        if self.offset:
            norm += self.bias
        return norm
    
    def __repr__(self):
        return f"{self.__class__.__name__}(shape = {self.shape},\
              eps={self.eps}, gamma={self.gamma}, offset={self.offset})"
