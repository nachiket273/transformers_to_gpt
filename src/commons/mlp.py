"""
Implements the MLP layer
"""
from torch import Tensor, randn
from torch.nn import Dropout, Linear, Module, ReLU
from torch.nn.init import normal_, xavier_normal_

class MLP(Module):
    r""" Multi Layer Perceptron Layer in the transformer.

    Args:
        input_features(int, required): Number of input features
        hidden_features(int):          Number of hidden features
                                       When not provided, hidden_features is set to
                                       4 * input_features
                                       Default: None
        out_features(int):             Number of output features
                                       When not orovided, out_features is set to input_features
                                       Default: None
        bias(bool):                    the additive bias
                                       if set to False, the layer will not learn an additive bias
                                       Default: True
        actn:                          Activation function
                                       Dafault: torch.nn.ReLU
        dropout_ratio(float):          Dropout ratio
                                       Default: 0.1
        inplace(bool):                 Whether to perform operation inplace
                                       Default: True

    Example:
        >>> mlp = MLP(4)
        >>> x = torch.randn((5, 4))
        >>> output = mlp(x)
    """
    def __init__(self, input_features: int, hidden_features: int = None, out_features: int = None,
                 bias: bool = True, actn= ReLU, dropout_ratio: float = 0.1, inplace: bool = True ) -> None:
        super().__init__()

        if not hidden_features or hidden_features < input_features:
            hidden_features = input_features * 4

        if not out_features or out_features < 0:
            out_features = input_features

        self.bias = bias
        self.fc1 = Linear(input_features, hidden_features, bias=self.bias)
        self.actn = actn(inplace=inplace)
        self.fc2 = Linear(hidden_features, out_features, bias=self.bias)
        self.dropout = Dropout(dropout_ratio, inplace=inplace)
        self.__init_weights__()
    
    def __init_weights__(self) -> None:
        xavier_normal_(self.fc1.weight)
        xavier_normal_(self.fc2.weight)
        if self.bias:
            normal_(self.fc1.bias)
            normal_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.actn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_features={self.fc1.in_features}, \
hidden_features={self.fc1.out_features}, out_features={self.fc2.out_features}, \
bias={self.bias}, actn={self.actn.__class__.__name__}, dropout_ratio={self.dropout.p}, \
inplace={self.dropout.inplace})"
