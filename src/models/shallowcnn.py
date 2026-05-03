import torch
import torch.nn as nn

class Square(nn.Module):
    def forward(self, x):
        return x ** 2

class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))

class ShallowConvNet(nn.Module):
    def __init__(self, number_channel=22, nb_classes=4, dropout_rate=0.5, time_kernel=25):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, time_kernel), (1, 1)),
            nn.Conv2d(40, 40, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            # Replace ELU with Square → AvgPool → SafeLog as in the original paper.
            Square(),
            nn.AvgPool2d((1, 75), (1, 15)),
            SafeLog(),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(out_features=nb_classes)

    def forward(self, x):
        x = self.shallownet(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
