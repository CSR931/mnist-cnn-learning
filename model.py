import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)   # normalisation

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected
        self.fc1 = nn.Linear(64*7*7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [32,14,14]
        # 第二卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [64,7,7]

        x = x.view(-1, 64*7*7)      # 展平为64*7*7=3136维
        x = F.relu(self.fc1(x))     #3136->128
        x = self.dropout(x)         # 随机丢弃50%
        x = self.fc2(x)             # 128->10
        return x
