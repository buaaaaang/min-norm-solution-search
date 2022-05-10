import torch
from torch import nn

def Model(s):
    if s == 'simple':
        return SimpleModel()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.optimizer = None
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(1, 1, False))
        self.layer_list.append(nn.Linear(1, 1, False))

    def forward(self, x):
        x = self.layer_list[0](x)
        x = self.layer_list[1](x)
        return x
    
    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def step(self):
        self.optimizer.zero_grad()
        output = self(torch.tensor([1.0]))
        loss = torch.square(output - 1)
        loss.backward()
        self.optimizer.step()
        return loss[0]

