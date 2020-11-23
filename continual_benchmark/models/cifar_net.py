import torch.nn as nn
import torch.nn.functional as F


class CifarNet(nn.Module):
    def __init__(self, **kw):
        super(CifarNet, self).__init__()
        self.n_channels = kw.get("n_channels") or 1
        self.n_classes = kw.get("n_classes") or 10
        self.d = kw.get("d") or 64
        self.in_size = kw.get("in_size") or 28
        self.model_bn = bool(int(kw.get("model_bn"))) or True
        self.n_conv = kw.get("n_conv") or 4
        self.max_pool = bool(int(kw.get("max_pool"))) or False
        self.p = kw.get("droput_rate") or 0.4
        if self.n_conv == 3:
            self.conv_out_dim = self.in_size//8
        else:
            self.conv_out_dim = self.in_size // 4

        if self.max_pool:
            self.conv_out_dim = self.conv_out_dim//4
        self.bn_0 = nn.BatchNorm2d(self.n_channels)
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=self.d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.d)
        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.dropout_1 = nn.Dropout(self.p)
        self.conv2 = nn.Conv2d(self.d, self.d*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(self.d*2)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.dropout_2 = nn.Dropout(self.p)
        self.conv3 = nn.Conv2d(self.d*2, self.d*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.d*4)
        self.dropout_3 = nn.Dropout(self.p)
        if self.n_conv == 3:
            self.fc3 = nn.Linear(self.d*4*self.conv_out_dim*self.conv_out_dim, 128)
        else:
            self.fc3 = nn.Linear(self.d * 2 * self.conv_out_dim * self.conv_out_dim, 128)
        self.dropout_4 = nn.Dropout(self.p)
        self.fc4 = nn.Linear(128, 64)
        self.last = nn.Linear(64, self.n_classes)

    def features(self, x):
        x = self.conv1(x)
        if self.model_bn:
            x = self.bn_1(x)
        if self.max_pool:
            x = self.max_pool_1(x)
        x = F.leaky_relu(x)
        x = self.dropout_1(x)
        x = self.conv2(x)
        if self.model_bn:
            x = self.bn_2(x)
        if self.max_pool:
            x = self.max_pool_2(x)
        x = F.leaky_relu(x)
        x = self.dropout_2(x)
        if self.n_conv > 2:
            x = self.conv3(x)
            if self.model_bn:
                x = self.bn_3(x)
            x = F.leaky_relu(x)
            x = self.dropout_3(x)
            x = x.view([-1, self.d*4*self.conv_out_dim*self.conv_out_dim])
        else:
            x = x.view([-1, self.d * 2 * self.conv_out_dim * self.conv_out_dim])
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def CifarNetG(**kw):
    return CifarNet(**kw)
