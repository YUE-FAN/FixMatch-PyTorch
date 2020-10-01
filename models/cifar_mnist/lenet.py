import torch.nn as nn
import torch
import numpy as np


class LeNet5(nn.Module):
    """
    100% MNIST: 99.25%, 99.23% 99.28% 99.08%
    80% MNIST: 99.16% 99.28% 99.12%
    60% MNIST: 99.05% 98.95%
    40% MNIST: 98.86%
    20% MNIST: 98.19%
    10% MNIST: 97.43%
    5% MNIST: 96.33%
    1% MNIST: 89.50%
    0.5% MNIST: 82.36%

    """
    def __init__(self, grayscale, num_classes):
        super(LeNet5, self).__init__()
        print("LeNet5 is used")
        if grayscale:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0, stride=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(84, num_classes)  # the last FC has to be defined as classifier across arch

    def forward(self, x):
        assert x.size(2) == 32 and x.size(3) == 32, "Input size of LeNet_5 has to be 32x32"
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        ret1 = x
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        ret2 = x
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.classifier(x)
        return [ret1, ret2, x]


def lenet5(**kwargs):
    """
    Constructs a LeNet5 model.
    """
    return LeNet5(**kwargs)