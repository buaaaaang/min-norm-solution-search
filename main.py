import torch
from torch import nn

from model import Model
from util import *
import sys

contraction = 0.9
termination = 0.9999
zero_loss = 0.01

model = Model('MNIST')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for iter in range(1000):
    print('try ', iter)
    # prepare contracted model with new optimizer
    torch.save({'model': model.state_dict()}, 'model.pth')
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict['model'])
    weight_contraction(model, contraction)
    model.set_optimizer(0.01)

    # train the network
    loss = zero_loss + 1
    angle = 0.0
    n_step = 0
    while (loss > zero_loss):
        n_step += 1
        loss = model.step(device)
        if loss <= zero_loss:
            angle = angle_of_gradient(model)
        sys.stdout.write("running steps: %d, loss: %.5f  \r" % (n_step, loss))
    print("runned steps: %d, final loss: %.5f" % (n_step, loss))
    print("angle: %.5f, weight_sum: %.5f" % (angle, norm_of_weight(model)))
    # print('try', iter+1, ': ', torch.cat([param.view(-1) for param in model.parameters()]))
    model.test(device)
    if angle > termination:
        print('angle got close enough.')
        break


