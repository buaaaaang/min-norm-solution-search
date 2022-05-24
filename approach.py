from util import *
import torch

contraction = 0.98
descent_rate = 5


def Approach(kind):
    if kind == 'contraction':
        return weight_contraction_optimizer(contraction)
    if kind == 'vertical_descent':
        return vertical_gradient_optimizer(descent_rate)


class weight_contraction_optimizer():
    def __init__(self, _contraction):
        self.contraction = _contraction

    def step(self, model):
        for i in range(len(model.layer_list)):
            model.layer_list[i].weight = torch.nn.Parameter(
                model.layer_list[i].weight * contraction)


class vertical_gradient_optimizer():
    def __init__(self, _descent_rate):
        self.descent_rate = _descent_rate
        self.n_step = 1

    def step(self, model):
        shape = []
        param_index = [0]
        param = []
        grad = []

        for i in range(len(model.layer_list)):
            layer = model.layer_list[i].weight
            shape.append(layer.size())
            param_index.append(tensor_size(layer) + param_index[i])
            param.append(layer.view(-1))
            grad.append(layer.grad.view(-1))

        param = torch.cat(param)
        grad = torch.cat(grad)

        direction = param - (torch.inner(grad, param) /
                             torch.inner(grad, grad)) * grad
        param = param - direction * self.descent_rate / \
            self.n_step / torch.norm(direction)
        self.n_step += 1

        for i in range(len(model.layer_list)):
            model.layer_list[i].weight = torch.nn.Parameter(
                param[param_index[i]:param_index[i+1]].view(shape[i])
            )
