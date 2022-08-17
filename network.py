import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 14)
        self.fc2 = nn.Linear(28 * 14, 28)
        self.fc3 = nn.Linear(28, 14)
        self.fc4 = nn.Linear(14, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(100, 78)
        self.fc2 = nn.Linear(78, 54)
        self.fc3 = nn.Linear(54, 36)
        self.fc4 = nn.Linear(36, 20)
        self.fc5 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
