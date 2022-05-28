import torch


def angle_of_gradient(model):
    param = torch.cat([layer.weight.view(-1) for layer in model.layer_list])
    grad = torch.cat([layer.weight.grad.view(-1) for layer in model.layer_list])
    return abs(torch.inner(param, grad).item() / ((param.pow(2).sum() * grad.pow(2).sum()).pow(0.5).item()))


def norm_of_weight(model):
    param = torch.cat([param.view(-1) for param in model.parameters()])
    return param.pow(2).sum().pow(0.5).item()

def norm_of_weight_for_student(model):
    l1 = model.layer_list[0].weight
    l2 = model.layer_list[1].weight
    norm = 0
    for i in range(l1.shape[0]):
        norm += l1[i,:].pow(2).sum().pow(0.5).item() * abs(l2[0,i])
    return norm


def tensor_size(t):
    shape = t.shape
    size = 1
    for i in range(len(shape)):
        size = size * shape[i]
    return size
