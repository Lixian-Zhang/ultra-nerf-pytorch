import torch

def cyclically_shift_dims_left(tensor: torch.Tensor):
    order = tuple(range(1, len(tensor.shape))) + (0, )
    return tensor.permute(order)

def merge_last_two_dims(tensor: torch.Tensor):
    new_shape = tensor.shape[:-2] + (-1, )
    return tensor.reshape(new_shape)