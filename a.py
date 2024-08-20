import torch

class Test(torch.nn.Module):

    def __init__(self, out_size):
        super().__init__()
        self.mlp_out = torch.nn.Parameter(torch.zeros(out_size, requires_grad=True))

    def forward(self, _):
        out = torch.empty_like(self.mlp_out)
        out[:, :, 0] = torch.abs(self.mlp_out[:, :, 0])
        out[:, :, 1:] = torch.sigmoid(self.mlp_out[:, :, 1:])
        return out


