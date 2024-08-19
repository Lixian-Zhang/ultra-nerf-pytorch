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

def add_a_leading_one(tensor: torch.Tensor, dim=0):
    dim %= len(tensor.shape)
    one = torch.ones_like(torch.index_select(tensor, dim=dim, index=torch.tensor([0], dtype=torch.long)))
    return torch.concat([one, tensor], dim=dim)

def repeat_last_element(tensor: torch.Tensor, dim=0):
    dim %= len(tensor.shape)
    last_element = torch.index_select(tensor, dim=dim, index=torch.tensor([tensor.shape[dim] - 1], dtype=torch.long))
    return torch.concat([tensor, last_element], dim=dim)

def plot_points(points, ref=None):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 300
    points = points.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if ref is not None:
        ref = ref.reshape(-1, 3)
        ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], c='r', marker='x')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.savefig('points.png')

def test():
    a = torch.randint(0, 10, (3, 5))
    b = repeat_last_element(a)
    c = add_a_leading_one(a, -1)
    print(a)
    print(b)
    print(c)


if __name__ == '__main__':
    test()