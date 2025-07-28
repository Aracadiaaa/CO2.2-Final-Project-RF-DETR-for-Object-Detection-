import torch.nn as nn

class RFEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.relu(self.conv3(x))
        x5 = self.relu(self.conv5(x))
        x7 = self.relu(self.conv7(x))
        return (x3 + x5 + x7) / 3
