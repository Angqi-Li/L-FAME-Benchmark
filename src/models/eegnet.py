import torch

from torch import nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, kernel_size=125, number_channel=22, signal_length=1000, 
                 pooling_size1=8, pooling_size2=8, f1=8, 
                 D=2, f2=16, norm_rate=0.25, dropout_rate=0.5):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Conv2d(1, f1, (1, kernel_size), (1,1), padding='same', bias=False) 
        self.batch_norm1 = nn.BatchNorm2d(f1)

        self.depthwise_conv = nn.Conv2d(f1, f1*D, (number_channel, 1), (1, 1), groups=f1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(f1*D)
        self.elu = nn.ELU()

        self.avg_pool1 = nn.AvgPool2d((1, pooling_size1))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.separable_conv = nn.Conv2d(f1*D, f2, (1, 16), padding='same', groups=f1*D, bias=False)

        self.seperable_conv_1x1 = nn.Conv2d(f2, f2, (1, 1), padding='same', bias=False)
        self.batch_norm3 = nn.BatchNorm2d(f2)

        self.avg_pool2 = nn.AvgPool2d((1, pooling_size2))
        self.dropout2 = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(f2 * (signal_length // pooling_size1 // pooling_size2), nb_classes) #f2 * (signal_length // pooling_size1 // pooling_size2)
        self.norm_rate = norm_rate

        # Register forward hooks to apply max_norm constraint
        self.depthwise_conv.register_forward_pre_hook(self.apply_max_norm_depthwise)
        self.classifier.register_forward_pre_hook(self.apply_max_norm_classifier)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        x = self.separable_conv(x)
        x = self.seperable_conv_1x1(x)
        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x
    
    def apply_max_norm_depthwise(self, module, input):
        with torch.no_grad():
            norm = self.depthwise_conv.weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=1.0)
            scale = desired / (norm + 1e-8)
            self.depthwise_conv.weight.data *= scale

    def apply_max_norm_classifier(self, module, input):
        with torch.no_grad():
            norm = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            scale = desired / (norm + 1e-8)
            self.classifier.weight.data *= scale