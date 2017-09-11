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


class VGGA(NN):

    def __init__(self, category_num=10):
        super(VGGA, self).__init__()
        self.conv1 = Conv(3, 64, 3, 1, 1)
        self.conv2 = Conv(64, 128, 3, 1, 1)
        self.conv3_1 = Conv(128, 256, 3, 1, 1)
        self.conv3_2 = Conv(256, 256, 3, 1, 1)
        self.conv4_1 = Conv(256, 512, 3, 1, 1)
        self.conv4_2 = Conv(512, 512, 3, 1, 1)
        self.conv5_1 = Conv(512, 512, 3, 1, 1)
        self.conv5_2 = Conv(512, 512, 3, 1, 1)
        self.fc1 = Conv(512, 4096, 7, 1, 0)
        self.fc2 = Conv(4096, 4096, 1, 1, 0)
        self.fc3 = Conv(4096, category_num, 1, 1, 0)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.name = 'vgg_a_{}'.format(category_num)

    def weight_initialization(self):
        self.conv1.weight_initialization(), self.conv2.weight_initialization()
        self.conv3_1.weight_initialization(), self.conv3_2.weight_initialization()
        self.conv4_1.weight_initialization(), self.conv4_2.weight_initialization()
        self.conv5_1.weight_initialization(), self.conv5_2.weight_initialization()
        self.fc1.weight_initialization(), self.fc2.weight_initialization(), self.fc3.weight_initialization()

    def __call__(self, x, train=False):
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, (2, 2), (2, 2), (0, 0))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, (2, 2), (2, 2), (0, 0))
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pool2d(h, (2, 2), (2, 2), (0, 0))
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.max_pool2d(h, (2, 2), (2, 2), (0, 0))
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.max_pool2d(h, (2, 2), (2, 2), (0, 0))
        h = self.drop1(h)
        h = F.relu(self.fc1(h))
        h = self.drop2(h)
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        h = self.global_average_pooling(h)
        return h
