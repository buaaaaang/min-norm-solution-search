import torch
from torch import nn

from model import Model
from util import *

contraction = 0.9
termination = 0.9999
zero_loss = 0.0001

model = Model('simple')

# initialize weight
nn.init.constant_(model.layer_list[0].weight, 0.1)
nn.init.constant_(model.layer_list[1].weight, 10.0)

for iter in range(100):
    # prepare contracted model with new optimizer
    torch.save({'model': model.state_dict()}, 'model.pth')
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict['model'])
    weight_contraction(model, contraction)
    model.set_optimizer()

    # train the network
    loss = zero_loss + 1
    angle = 0.0
    while (loss > zero_loss):
        loss = model.step()
        if loss <= zero_loss:
            angle = angle_of_gradient(model)

    print('try', iter+1, ': ', torch.cat([param.view(-1) for param in model.parameters()]))
    if angle > termination: break


