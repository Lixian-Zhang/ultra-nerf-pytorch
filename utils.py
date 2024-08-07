import torch
from torch.nn.functional import pad

def cyclically_shift_dims_left(tensor: torch.Tensor):
    order = tuple(range(1, len(tensor.shape))) + (0, )
    return tensor.permute(order)

def add_a_leading_zero(tensor: torch.Tensor):
    if len(tensor.shape) == 1:
        return pad(tensor, pad=(1, 0), mode='constant', value=0)
    elif len(tensor.shape) == 2:
        return pad(tensor, pad=(0, 0, 1, 0), mode='constant', value=0)

def add_a_leading_one(tensor: torch.Tensor):
    if len(tensor.shape) == 1:
        return pad(tensor, pad=(1, 0), mode='constant', value=1)
    elif len(tensor.shape) == 2:
        return pad(tensor, pad=(0, 0, 1, 0), mode='constant', value=1)


def plot_points(points):
    import matplotlib.pyplot as plt
    points = points.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], )
    fig.savefig('points.png')

def test():
    a = torch.ones(3)
    print(add_a_leading_zero(a))
    pass


if __name__ == '__main__':
    test()