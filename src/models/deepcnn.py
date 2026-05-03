import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, number_channel=22, nb_classes=4, dropout_rate=0.5, time_kernel=10):
        super().__init__()

        self.deepet = nn.Sequential(
            # First layer uses a configurable temporal kernel length.
            nn.Conv2d(1, 25, (1, time_kernel), (1, 1)),
            nn.Conv2d(25, 25, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1,3), (1,3)),
            nn.Dropout(dropout_rate),

            # Deeper layers keep kernel=10: after MaxPool the temporal resolution is
            # already compressed, so a fixed-size kernel spans a much longer context.
            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1,3), (1,3)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1,3), (1,3)),
            nn.Dropout(dropout_rate),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1,3), (1,3)),
            nn.Dropout(dropout_rate),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(out_features=nb_classes)

    def forward(self, x):
        x = self.deepet(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
