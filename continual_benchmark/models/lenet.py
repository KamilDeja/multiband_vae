import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, **kw):
        super(LeNet, self).__init__()
        self.p = 0.6
        self.n_channels = kw.get("n_channels") or 1
        self.n_classes = kw.get("n_classes") or 10
        self.d = kw.get("d") or 64
        self.in_size = kw.get("in_size") or 28
        self.conv_out_dim = self.in_size//8
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d)
        self.dropout_1 = nn.Dropout(self.p)
        self.conv2 = nn.Conv2d(self.d, self.d*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d*2)
        self.dropout_2 = nn.Dropout(self.p)
        self.conv3 = nn.Conv2d(self.d*2, self.d*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d*4)
        self.dropout_3 = nn.Dropout(self.p)
        self.fc3 = nn.Linear(self.d*4*self.conv_out_dim*self.conv_out_dim, 84)
        self.dropout_4 = nn.Dropout(self.p)
        self.last = nn.Linear(84,self.n_classes)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn_1(x)
        x = F.leaky_relu(x)
        x = self.dropout_1(x)
        x = self.conv2(x)
        x = self.bn_2(x)
        x = F.leaky_relu(x)
        x = self.dropout_2(x)
        x = self.conv3(x)
        x = self.bn_3(x)
        x = F.leaky_relu(x)

        x = self.dropout_3(x)
        x = x.view([-1,self.d*4*self.conv_out_dim*self.conv_out_dim])
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout_4(x)
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def LeNetG(**kw):  # LeNet with grey input
    return LeNet(**kw)
