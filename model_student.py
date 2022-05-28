import torch
from torch import nn
from model_base import BaseModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.layer1 = nn.Linear(2, 3, False)
        self.layer2 = nn.Linear(3, 1, False)
        w1 = self.layer1.weight.detach().numpy()
        with torch.no_grad():
            for i in range(3):
                self.layer2.weight[0, i] = self.layer2.weight[0, i] / (
                    (w1[i][0] ** 2 + w1[i][1] ** 2) ** 0.5 * abs(self.layer2.weight[0, i]))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)


class Student(BaseModel):
    def __init__(self):
        super(Student, self).__init__()
        self.teacher = Teacher()
        teacher_state_dict = torch.load('teacher_bad_regularization.pth')
        self.teacher.load_state_dict(teacher_state_dict['model'])
        self.n_train_data = 15 #200 #15
        with torch.no_grad():
            self.train_input = nn.functional.normalize(
                torch.randn((self.n_train_data, 2)))
            self.train_output = self.teacher(self.train_input)
        self.n_hidden_nodes = 100 #200 #20
        self.layer_list.append(nn.Linear(2, self.n_hidden_nodes, False))
        self.layer_list.append(nn.Linear(self.n_hidden_nodes, 1, False))
        for layer in self.layer_list:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.5)

    def forward(self, x):
        x = F.relu(self.layer_list[0](x))
        return self.layer_list[1](x)

    def step(self, device):
        self.optimizer.zero_grad()
        output = self(self.train_input)
        loss = F.mse_loss(output, self.train_output)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self, device):
        # population loss
        n_test_data = 100000
        with torch.no_grad():
            test_input = nn.functional.normalize(torch.randn((n_test_data, 2)))
            test_output = self.teacher(test_input)
            output = self(test_input)
            loss = F.mse_loss(output, test_output)
        return loss.item()

    def draw_weights(self):
        plt.close()
        parent_weight1 = self.teacher.layer1.weight.detach().numpy()
        parent_weight2 = self.teacher.layer2.weight.detach().numpy()
        for i in range(parent_weight1.shape[0]):
            plt.plot([0, parent_weight1[i][0] * abs(parent_weight2[0][i])],
                     [0, parent_weight1[i][1] * abs(parent_weight2[0][i])], color="black")
        student_weight1 = self.layer_list[0].weight.detach().numpy()
        student_weight2 = self.layer_list[1].weight.detach().numpy()
        enlarge = self.n_hidden_nodes / 30.
        for i in range(student_weight1.shape[0]):
            if student_weight2[0][i] > 0:
                plt.scatter(student_weight1[i][0] * abs(student_weight2[0][i]) * enlarge,
                            student_weight1[i][1] *
                            abs(student_weight2[0][i]) * enlarge,
                            color='red', s=5)
            else:
                plt.scatter(student_weight1[i][0] * abs(student_weight2[0][i]) * enlarge,
                            student_weight1[i][1] *
                            abs(student_weight2[0][i]) * enlarge,
                            color='blue', s=5)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.rcParams["figure.figsize"] = (5,5)
        plt.show(block=False)
        plt.pause(0.3)


if __name__ == '__main__':
    teacher = Teacher()
    torch.save({'model': teacher.state_dict()}, 'teacher.pth')

    model = Student()
    model.draw_weights()
