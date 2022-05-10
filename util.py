import torch

def weight_contraction(model, alpha):
    for i in range(len(model.layer_list)):
        model.layer_list[i].weight = torch.nn.Parameter(model.layer_list[i].weight * alpha)

def angle_of_gradient(model):
    param = torch.cat([param.view(-1) for param in model.parameters()])
    grad = torch.cat([param.grad.view(-1) for param in model.parameters()])
    return abs(torch.inner(param, grad).item() / ((param.pow(2).sum() * grad.pow(2).sum()).pow(0.5).item()))