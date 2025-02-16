import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2)
        self.fc1 = nn.Linear(7*7*16, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 7*7*16)
        x = self.fc1(x)
        return x