import torch
from torch import nn
        
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.optimizer = None
        self.layer_list = nn.ModuleList()

    def set_optimizer(self, lr=0.001, type='SGD'):
        if (type == 'SGD'):
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif (type == 'ADAM'):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            print(type, 'is not supported')
            raise ValueError()

    def step(self, device):
        raise NotImplementedError()

    def test(self, device):
        raise NotImplementedError()

class SimpleModel(BaseModel):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer_list.append(nn.Linear(1, 1, False))
        self.layer_list.append(nn.Linear(1, 1, False))
        nn.init.constant_(self.layer_list[0].weight, 0.1)
        nn.init.constant_(self.layer_list[1].weight, 10.0)

    def forward(self, x):
        x = self.layer_list[0](x)
        x = self.layer_list[1](x)
        return x

    def step(self, device):
        self.optimizer.zero_grad()
        output = self(torch.tensor([1.0]))
        loss = torch.square(output - 1)
        loss.backward()
        self.optimizer.step()
        return loss[0]

    def test(self, device):
        return 1


