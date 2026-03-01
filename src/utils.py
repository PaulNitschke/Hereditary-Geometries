import torch as th
import torch
    
class DenseNN(th.nn.Module):
    """A fully connected neural network with arbitrary layer sizes and ReLU activations."""
    def __init__(self, layer_sizes: list[int]):
        """
        Args:
            layer_sizes (list of int): List of layer sizes, including input and output dimensions.
                                       Example: [2, 64, 128, 2]
        """
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(th.nn.Linear(in_dim, out_dim))
            layers.append(th.nn.ReLU()) 
        layers.pop()
        self.net = th.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

# class FrameNet(th.nn.Module):
#     def __init__(self, ambient_dim: int, kernel_dim: int, hidden_layer_dims: list[int]):
#         """
#         Dense neural network to approximate a basis of shape kernel_dim x ambient_dim.
#         Returns a tensor of shape (kernel_dim, ambient_dim) representing the basis.
#         Includes an affine translation term in the last layer.
#         """
#         super().__init__()
#         assert kernel_dim == 1, "Only one-dimensional kernel tested."

#         layers = []
#         in_dim = ambient_dim

#         # Hidden layers (optional)
#         for h_dim in hidden_layer_dims:
#             layers.append(th.nn.Linear(in_dim, h_dim))
#             layers.append(th.nn.ReLU())
#             in_dim = h_dim

#         self.last_linear = th.nn.Linear(in_dim, ambient_dim * kernel_dim, bias=True)

#         self.net = th.nn.Sequential(*layers)
#         self.ambient_dim, self.kernel_dim = ambient_dim, kernel_dim

#     def forward(self, x):
#         flat = self.net(x)
#         flat = self.last_linear(flat)
#         M = flat.view(-1, self.kernel_dim, self.ambient_dim)
#         return M

class SingleLayerNet(torch.nn.Module):
    def __init__(self, ambient_dim, kernel_dim, oracle_linear=None, oracle_bias=None):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.kernel_dim = kernel_dim
        self.linear = torch.nn.Linear(ambient_dim, kernel_dim * ambient_dim, bias=True)

        # Optional oracle initialization
        if oracle_linear is not None and oracle_bias is not None:
            with torch.no_grad():
                self.linear.weight.copy_(oracle_linear)
                self.linear.bias.copy_(oracle_bias)

    def forward(self, x):
        x = self.linear(x)
        return x.view(-1, self.kernel_dim, self.ambient_dim)
    

# class FrameMatrixNet(th.nn.Module):
#     def __init__(self, ambient_dim: int, null_dim: int, hidden_layer_dims: list[int]):
#         """
#         Dense NN to approximate a basis of shape (null_dim, ambient_dim, ambient_dim).
#         Uses QR decomposition to ensure orthonormal outputs.
#         """
#         super().__init__()
#         self.ambient_dim = ambient_dim
#         self.null_dim = null_dim

#         in_dim = ambient_dim
#         layers = []
#         for h_dim in hidden_layer_dims:
#             layers.append(th.nn.Linear(in_dim, h_dim))
#             layers.append(th.nn.ReLU())
#             in_dim = h_dim
#         layers.append(th.nn.Linear(in_dim, ambient_dim * ambient_dim * null_dim))
#         self.net = th.nn.Sequential(*layers)

#     def forward(self, x):
#         # Originally had a QR decomposition here, but it was unstable.
#         flat = self.net(x)
#         M = flat.view(-1, self.null_dim, self.ambient_dim, self.ambient_dim)
#         return M

    

def get_non_default_args(parser, parsed_args) -> dict:
    """Returns all non-default arguments from the parsed args."""
    defaults = parser.parse_args([])
    return {
    k: v for k, v in vars(parsed_args).items()
    if getattr(parsed_args, k) != getattr(defaults, k)}