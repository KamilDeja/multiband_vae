from torch.nn import Module
from torch import nn


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        dropout_rate = 0.4
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(64, 16, 5)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.LeakyReLU()
        # self.pool2 = nn.MaxPool2d(2)
        self.dropout_3 = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(256, 256)
        self.relu3 = nn.LeakyReLU()
        self.dropout_4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.LeakyReLU()
        self.dropout_5 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 20)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.bn1(y)
        y = self.dropout_1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
#         y = self.pool2(y)
        y = self.dropout_2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.dropout_3(y)

        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.dropout_4(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.dropout_5(y)
        y = self.fc3(y)
        # y = torch.softmax(y, dim=1)
        return y

    ##### Part forward without last classification layer for the purpose of FID computing
    def part_forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)

        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        # y = torch.softmax(y, dim=1)
        return y
