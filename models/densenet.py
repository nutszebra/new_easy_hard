import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prototype import NN


class BN_ReLU_Conv(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad)
        self.bn = nn.BatchNorm2d(in_channel)

    def weight_initialization(self):
        self.conv.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv))
        self.conv.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv, constant=0))

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))


class Transition_Layer(NN):

    def __init__(self, in_channel, out_channel):
        super(Transition_Layer, self).__init__()
        self.bn_relu_conv = BN_ReLU_Conv(in_channel, out_channel, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))

    def weight_initialization(self):
        self.bn_relu_conv.weight_initialization()

    def forward(self, x):
        return F.avg_pool2d(self.bn_relu_conv(x), (2, 2), (2, 2), 0)


class DenseBlock(NN):

    def __init__(self, in_channel, block_size=16, growth_rate=12, multiplier=4):
        super(DenseBlock, self).__init__()
        for i in six.moves.range(block_size):
            self['bn_relu_conv{}_1'.format(i)] = BN_ReLU_Conv(in_channel, growth_rate * multiplier, 1, 1, 0)
            self['bn_relu_conv{}_2'.format(i)] = BN_ReLU_Conv(growth_rate * multiplier, growth_rate, 3, 1, 1)
            in_channel = in_channel + growth_rate
        self.block_size = block_size

    def weight_initialization(self):
        for i in six.moves.range(self.block_size):
            self['bn_relu_conv{}_1'.format(i)].weight_initialization()
            self['bn_relu_conv{}_2'.format(i)].weight_initialization()

    def forward(self, x):
        for i in six.moves.range(self.block_size):
            h = self['bn_relu_conv{}_1'.format(i)](x)
            h = self['bn_relu_conv{}_2'.format(i)](h)
            x = torch.cat((x, h), 1)
        return x


class DenselyConnectedCNN(NN):

    def __init__(self, category_num, block_num=3, block_size=32, growth_rate=12):
        super(DenselyConnectedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        in_channel = 16
        for i in six.moves.range(block_num):
            self['dense{}'.format(i)] = DenseBlock(in_channel, block_size, growth_rate)
            in_channel = in_channel + growth_rate * block_size
            # if block_num=3, then trans1 and trans2 are used
            if i <= block_num - 1:
                self['trans{}'.format(i)] = Transition_Layer(in_channel, int(in_channel * 0.5))
                in_channel = int(in_channel * 0.5)
        self['bn1'] = nn.BatchNorm2d(in_channel)
        self['fc1'] = nn.Linear(in_channel, category_num)
        self.block_num = block_num
        self.name = 'densenet_{}_{}_{}_{}'.format(category_num, block_num, block_size, growth_rate)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.fc1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.fc1))
        self.fc1.bias.data = torch.FloatTensor(NN.bias_initialization(self.fc1, constant=0))
        for i in six.moves.range(self.block_num):
            self['dense{}'.format(i)].weight_initialization()
            if i <= self.block_num - 1:
                self['trans{}'.format(i)].weight_initialization()

    def forward(self, x):
        h = self.conv1(x)
        for i in six.moves.range(self.block_num):
            h = self['dense{}'.format(i)](h)
            if i <= self.block_num - 1:
                h = self['trans{}'.format(i)](h)
        h = F.relu(self.bn1(h))
        h = self.global_average_pooling(h)
        h = self.fc1(h)
        return h
