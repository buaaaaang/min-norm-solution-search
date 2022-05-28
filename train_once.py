import torch
from torch import nn
import sys

from util import *
from model import Model
from approach import Approach

zero_loss = 0.00001

model = Model('student')
approach = Approach('vertical_descent') #Approach('contraction') #Approach('vertical_descent')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

approach.step(model)
model = model.to(device)
model.set_optimizer(0.1, decay=0.0008)

# train the network
loss = zero_loss + 1
angle = 0.0
n_step = 0
while (True):
    n_step += 1
    loss = model.step(device)
    sys.stdout.write(
        "epoch %d, running steps: %d, loss: %.9f \r" % (n_step, n_step, loss))
    if (n_step % 1000 == 0):
        angle = angle_of_gradient(model)
        test = model.test(device)
        print("epoch %d, run_steps: %d, train_loss: %.9f," % (n_step, n_step, loss),
                "squared_weight_sum: %.5f, test_loss: %.9f, angle: %.5f" % (norm_of_weight(model)**2, test, angle))
        model.draw_weights()
