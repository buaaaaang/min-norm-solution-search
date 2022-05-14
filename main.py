import torch
from torch import nn

from model import Model
from util import *
import sys

contraction = 0.99
termination = 0.9999
zero_loss = 0.000001

model = Model('MNIST')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for iter in range(1000):
    # prepare contracted model with new optimizer
    torch.save({'model': model.state_dict()}, 'model.pth')
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict['model'])
    weight_contraction(model, contraction)
    model = model.to(device)
    model.set_optimizer(0.1)

    # train the network
    loss = zero_loss + 1
    angle = 0.0
    n_step = 0
    while (loss > zero_loss):
        n_step += 1
        loss = model.step(device)
        if loss <= zero_loss:
            angle = angle_of_gradient(model)
        sys.stdout.write("try %d, running steps: %d, loss: %.9f  \r" % (iter, n_step, loss))
    test = model.test(device)
    print("try %d, runned_steps: %d, train_loss: %.9f, angle: %.5f," % (iter, n_step, loss, angle),
        "weight_sum: %.5f, test_loss: %.9f" % (norm_of_weight(model), model.test(device)))
    # print('try', iter+1, ': ', torch.cat([param.view(-1) for param in model.parameters()]))
    if angle > termination:
        print('angle got close enough.')
        break


