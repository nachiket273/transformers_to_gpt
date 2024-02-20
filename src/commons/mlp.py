"""
Implements the MLP layer
"""
from torch import Tensor, float32
from torch.nn import Dropout, Linear, Module, ReLU
from torch.nn.init import normal_, xavier_normal_

class MLP(Module):
    def __init__(self, input_features: int, hidden_features: int = None, out_features: int = None,
                 bias: bool = True, actn= ReLU, dropout_ratio: float = 0.1, device: str = 'cpu',
                 dtype = float32, inplace: bool = True ) -> None:
        super().__init__()

        if not hidden_features or hidden_features < input_features:
            hidden_features = input_features * 4

        if not out_features or out_features < 0:
            out_features = input_features

        self.fc1 = Linear(input_features, hidden_features, bias=bias, device=device, dtype=dtype)
        self.actn = actn()
        self.fc2 = Linear(hidden_features, out_features, bias=bias, device=device, dtype=dtype)
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

