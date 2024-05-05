"""
Author    Daeheon Kwon (2024)
Date      2024.05.04
"""

import torch
import torch.nn as nn
import numpy as np

class SciCNN(nn.Module):

    def __init__(self):
        super(SciCNN, self).__init__()        

        self.inception1 = Inception(8, 8, 16, 8, 8)
        self.maxpool1 = nn.MaxPool2d((1, 4), stride=(1, 4), ceil_mode=True)
        self.inception2 = Inception(16, 16, 8, 16, 4)
        self.maxpool2 = nn.MaxPool2d((1, 4), stride=(1, 4), ceil_mode=True)
        self.inception3 = Inception(32, 32, 4, 32, 2)
        self.maxpool3 = nn.MaxPool2d((1, 8), stride=(1, 8), ceil_mode=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(22*64, 16)
        self.npc = NPC()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.inception1(x)
        x = self.maxpool1(x)
        x = self.inception2(x)
        x = self.maxpool2(x)
        x = self.inception3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1, ch1_kernel, ch2, ch2_kernel):
        super(Inception, self).__init__()
        self.branch1 = BasicConv1d(in_channels, ch1, kernel=(1, ch1_kernel), padding=(0, (ch1_kernel-1)//2))
        self.branch2 = BasicConv1d(in_channels, ch2, kernel=(1, ch2_kernel), padding=(0, (ch2_kernel-1)//2))
        self.se = SEModule()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        concat = torch.cat([branch1, branch2], dim=1)
        return self.se(concat)
        
class SEModule(nn.Module):
    def __init__(self, channels=22, reduction=4):
        super(SEModule, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out = torch.squeeze(self.globalAvgPool(x))
        out = self.fc(out).view(x.size(0), x.size(1), 1, 1)
        out = x * out.expand_as(x)
        return out.permute(0, 3, 1, 2)
    
class BasicConv1d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel, padding):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=True),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                            )
    def forward(self, x):
        return self.conv(x)
    
class NPC(nn.Module):
    def __init__(self, num_clusters=256):
        super(NPC, self).__init__()
        # 256 predefined positions of NPC clusters
        self.position = nn.Parameter(torch.from_numpy(np.random.normal(0, 1.5, (num_clusters, 16, 1))).to(torch.float32), requires_grad=True)
        self.label = nn.Parameter(torch.ones(num_clusters), requires_grad=False)

