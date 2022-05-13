import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torchvision.transforms as transfroms


def Model(type, config=None):
    if type == 'simple':
        return SimpleModel()
    if type == 'MNIST':
        return MNISTClassifier()

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

class MNISTClassifier(BaseModel):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        train_data = datasets.MNIST(root = './', train=True, download=True, 
            transform = transfroms.Compose([transfroms.ToTensor()]))
        self.test_data = datasets.MNIST(root = './', train=False, download=True, 
            transform = transfroms.Compose([transfroms.ToTensor()]))
        self.train_data, _ = random_split(train_data, [60, len(train_data)-60])
        print('reduced train data from ', len(train_data), ' to ', len(self.train_data))
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=60, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=100, shuffle=True)
        
        self.layer_list.append(nn.Linear(784, 1024, False))
        self.layer_list.append(nn.Linear(1024, 10, False))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_list[0](x))
        x = self.layer_list[1](x)
        return x

    def step(self, device):
        avg_loss = 0
        for data, target in self.train_loader:
            data = data.to(device).float()
            target = target.to(device)
            target = F.one_hot(target, num_classes=10).float()
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()/len(self.train_loader)
        return avg_loss

    def test(self, device):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in self.test_loader:
                data = data.to(device).float()
                target = target.to(device)
                output = self(data)
                preds = torch.max(output.data, 1)[1]
                total += len(target)
                correct += (preds==target).sum().item()
                
            print('Test Accuracy: ', 100.*correct/total, '%')




    
if __name__=='__main__':
    MNISTClassifier({'len_train_data': 600})





