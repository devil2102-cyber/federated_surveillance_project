import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeCNNClassifier(nn.Module):
    """A lightweight CNN for edge diagnostics (e.g. infection classification)."""
    def __init__(self, num_classes=2):
        super(EdgeCNNClassifier, self).__init__()
        # Input channels = 3 (RGB), Output channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Output channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Output channels = 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # After 3 max pools of 2x2, a 224x224 image becomes 28x28
        # 64 channels * 28 * 28 = 50176
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
