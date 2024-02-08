import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input : batch, 224, 224, 3 (3은 RGB)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        # batch, 222, 222, 32
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        # batch, 111, 111, 32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # batch, 109, 109, 64
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        # batch, 54, 54, 64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # batch, 52, 52, 128
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        # batch, 26, 26, 128
        self.fc1 = nn.Linear(26*26*128, 80)
        self.fc2 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        x = x.view(-1, 26*26*128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
