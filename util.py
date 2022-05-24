import torch


def angle_of_gradient(model):
    param = torch.cat([param.view(-1) for param in model.parameters()])
    grad = torch.cat([param.grad.view(-1) for param in model.parameters()])
    return abs(torch.inner(param, grad).item() / ((param.pow(2).sum() * grad.pow(2).sum()).pow(0.5).item()))


def norm_of_weight(model):
    param = torch.cat([param.view(-1) for param in model.parameters()])
    return param.pow(2).sum().pow(0.5).item()


def tensor_size(t):
    shape = t.shape
    size = 1
    for i in range(len(shape)):
        size = size * shape[i]
    return size
