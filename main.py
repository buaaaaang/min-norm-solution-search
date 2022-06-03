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

for iter in range(100):
    approach.step(model)
    model = model.to(device)
    model.set_optimizer(0.1) #model.set_optimizer(0.1, decay=0.0007)

    # train the network
    loss = zero_loss + 1
    angle = 0.0
    n_step = 0
    while (loss > zero_loss):
        n_step += 1
        loss = model.step(device)
        if (loss <= zero_loss):
            angle = angle_of_gradient(model)
        sys.stdout.write(
            "try %d, running steps: %d, loss: %.9f \r" % (iter, n_step, loss))
    test = model.test(device)
    print("try %d, run_steps: %d, train_loss: %.9f," % (iter, n_step, loss),
          "squarted_weight_sum: %.5f, test_loss: %.9f, angle: %.5f" % (norm_of_weight(model)**2, test, angle))
    #print("norm of weight of student %.5f" % norm_of_weight_for_student(model))
    if (iter % 3 == 0):
        #model.draw_weights()
        model.save_weights(iter)
