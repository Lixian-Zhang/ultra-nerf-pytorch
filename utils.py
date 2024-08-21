import torch

def cyclically_shift_dims_left(tensor: torch.Tensor):
    order = tuple(range(1, len(tensor.shape))) + (0, )
    return tensor.permute(order)

def add_a_leading_zero(tensor: torch.Tensor, dim=0):
    dim %= len(tensor.shape)
    zero = torch.zeros_like(torch.index_select(tensor, dim=dim, index=torch.tensor([0], dtype=torch.long)))
    return torch.concat([zero, tensor], dim=dim)

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

def sample_bernoulli(probabilities_yielding_one):
    probabilities_yielding_zero = 1 - probabilities_yielding_one
    logits = torch.logit(torch.stack([probabilities_yielding_one, probabilities_yielding_zero], dim=-1), eps=1e-4)
    return torch.nn.functional.gumbel_softmax(logits, tau=1e-1, hard=True)[..., 0]

def test():
    p = torch.rand(2, 3, requires_grad=True)
    print(p)
    with torch.no_grad():
        frequency = torch.zeros_like(p)
        for _ in range(100):
            frequency += sample_bernoulli(p)
        print(p - frequency / 100.)

    s = sample_bernoulli(p)
    print(s)
    loss = torch.nn.functional.mse_loss(s, -1 * torch.ones_like(s))
    loss.backward()
    print(p.grad)


if __name__ == '__main__':
    test()