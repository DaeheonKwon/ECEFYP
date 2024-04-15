import torch
import torch.nn as nn

class SciCNN(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()        

        self.inception1 = Inception(8, 8, 16, 8, 8)
        self.maxpool1 = nn.MaxPool(4, stride=4, ceil_mode=True)
        self.inception2 = Inception(16, 16, 8, 16, 4)
        self.maxpool2 = nn.MaxPool(4, stride=8, ceil_mode=True)
        self.inception3 = Inception(32, 32, 4, 32, 2)
        self.maxpool3 = nn.MaxPool(8, stride=8. ceil_mode=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1408, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
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
        self.branch1 = BasicConv1d(in_channels, ch1, kernel_size=ch1_kernel)
        self.branch2 = BasicConv1d(in_channels, ch2, kernel_size=ch2_kernel)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], dim=3)

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, **kwargs),
                            nn.ReLU()
                            )
    def forward(self, x):
        return self.conv(x)