import pytorch_lightning
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def num_parameters(self):
        return self.num_parameters

    def forward(self, x):
        #B, H, W = 32, 28, 28
        x = F.relu(self.conv1(x)) # (B, 32, 28, 28)
        x = self.pool(x) #(B, 32, 14, 14)
        x = F.relu(self.conv2(x)) # (B, 64, 14, 14) 
        x = self.pool(x) # (B, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor, combine batches (B, 65)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
