import torch
import torch.nn as nn
import torch.nn.functional as F
from .prototype import NN


class Conv(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad)

    def weight_initialization(self):
        self.conv.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv))
        self.conv.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv, constant=0))

    def forward(self, x):
        return self.conv(x)


class Linear(NN):

    def __init__(self, in_channel, out_channel):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel)

    def weight_initialization(self):
        self.fc.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.fc))
        self.fc.bias.data = torch.FloatTensor(NN.bias_initialization(self.fc, constant=0))

    def forward(self, x):
        return self.fc(x)


class AlexNet(NN):

    def __init__(self, category_num=10):
        super(AlexNet, self).__init__()
        self.conv1 = Conv(3, 64, 11, 4, 2)
        self.conv2 = Conv(64, 192, 5, 1, 2)
        self.conv3 = Conv(192, 384, 3, 1, 1)
        self.conv4 = Conv(384, 256, 3, 1, 1)
        self.conv5 = Conv(256, 256, 3, 1, 1)
        self.fc6 = Linear(256 * 6 * 6, 4096)
        self.fc7 = Linear(4096, 4096)
        self.fc8 = Linear(4096, category_num)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.name = 'alexnet_{}'.format(category_num)

    def weight_initialization(self):
        self.conv1.weight_initialization(), self.conv2.weight_initialization()
        self.conv3.weight_initialization(), self.conv4.weight_initialization()
        self.conv5.weight_initialization(), self.fc6.weight_initialization()
        self.fc7.weight_initialization(), self.fc8.weight_initialization()

    def forward(self, x, train=False):
        h = F.max_pool2d(
            F.relu(self.conv1(x)), 3, 2, 0)
        h = F.max_pool2d(
            F.relu(self.conv2(h)), 3, 2, 0)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(F.relu(self.conv5(h)), 3, 2, 0)
        h = h.view(-1, 256 * 6 * 6)
        h = self.drop1(F.relu(self.fc6(h)))
        h = self.drop2(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h
