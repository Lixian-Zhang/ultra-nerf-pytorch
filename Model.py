import torch
from torch import nn
from torchvision.ops import MLP

from utils import cyclically_shift_dims_left, merge_last_two_dims

class NerualRadianceField(nn.Module):

    def __init__(self):
        super().__init__()
        self.positional_encoding_dim = 20
        self.query_dim  = 3 # 3 for xyz in 3D space
        self.output_dim = 5 # 5 element parameter vector for physics-inspired rendering, see: "Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging"
        self.width = 256
        self.depth = 8
        self.pe = PositionalEncoding(self.positional_encoding_dim)
        self.mlp_input_dim = self.positional_encoding_dim * self.query_dim # positional encoding is done separately on each query dimension
        self.mlp1 = MLP(
             self.mlp_input_dim, 
            [self.width for _ in range(self.depth // 2)]
        )
        self.mlp2 = MLP(
             self.mlp_input_dim + self.width, # skip connection in the middle
            [self.width for _ in range(self.depth // 2 - 1)] + [self.output_dim]
        )

    def forward(self, query):
        query = merge_last_two_dims(self.pe(query))
        return self.mlp2( torch.concat([query, self.mlp1(query)], dim=-1) )

class PositionalEncoding:
    # see formula (4) of paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", output_dim == 2 * L

    def __init__(self, output_dim: int): 
        self.two_exponents = torch.exp2(torch.arange(output_dim // 2)) # half sine and half cosine

    def __call__(self, p: torch.Tensor):
        if not torch.is_tensor(p):
            p = torch.tensor(p, dtype=torch.float)

        if len(p.shape) == 0:
            # scalar input
            intermediate = torch.pi * p * self.two_exponents
            return torch.concat( [torch.sin(intermediate), torch.cos(intermediate)] ) # the sin and cos are not interlaced
        else:
            # when dealing with batched input a new dimension of 2 * L will be appended
            intermediate_shape = (len(self.two_exponents), ) + p.shape
            intermediate = torch.pi * (self.two_exponents.reshape(-1, 1) @ p.reshape(1, -1)).reshape(intermediate_shape)
            return cyclically_shift_dims_left(torch.concat( [torch.sin(intermediate), torch.cos(intermediate)] ))

def test_nerf():
    nerf = NerualRadianceField()

    # test single input
    query = torch.ones(nerf.query_dim)
    out = nerf(query)
    assert out.shape == (nerf.output_dim, )

    # test batch
    batch_size = 16
    query = torch.ones(batch_size, nerf.query_dim)
    out = nerf(query)
    assert out.shape == (batch_size, nerf.output_dim)
    
def test_positional_encoding():
    L = 10
    pe = PositionalEncoding(2 * L)
    assert len(pe(0.1)) == 2 * L
    
    # test correct output shape
    p = 0.1 * torch.ones(64, 3)
    res = pe(p)
    assert res.shape == p.shape + (2 * L, )

    # test equavalence between batched computation and one-at-a-time computation
    for i in range(len(p)):
        assert torch.all(pe(p[i]) == res[i])
    for i in range(len(p[0])):
        assert torch.all(pe(p[0, i]) == res[0, i])

if __name__ == '__main__':
    test_nerf()
    test_positional_encoding()