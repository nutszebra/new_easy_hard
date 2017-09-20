import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prototype import NN


class Bridge(NN):

    def __init__(self, in_channel, pad, pool_flag=True):
        super(Bridge, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.pad = pad
        self.pool_flag = pool_flag

    def forward(self, x):
        x = self.bn(x)
        if self.pool_flag:
            x = F.avg_pool2d(x, 1, 2, 0)
        return self.concatenate_zero_pad(x, self.pad)


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


class ResBlock(NN):

    def __init__(self, in_channel, out_channel, bridge=NN(), n=18, stride_at_first_layer=2):
        super(ResBlock, self).__init__()
        for i in six.moves.range(n):
            self['bn_relu_conv1_{}'.format(i)] = BN_ReLU_Conv(in_channel, out_channel, 3, stride_at_first_layer, 1)
            self['bn_relu_conv2_{}'.format(i)] = BN_ReLU_Conv(out_channel, out_channel, 3, 1, 1)
            stride_at_first_layer = 1
            in_channel = out_channel
        self.bridge = bridge
        self.n = n

    def weight_initialization(self):
        self.bridge.weight_initialization()
        for i in six.moves.range(self.n):
            self['bn_relu_conv1_{}'.format(i)].weight_initialization()
            self['bn_relu_conv2_{}'.format(i)].weight_initialization()

    def forward(self, x):
        for i in six.moves.range(self.n):
            h = self['bn_relu_conv1_{}'.format(i)](x)
            h = self['bn_relu_conv2_{}'.format(i)](h)
            if i == 0:
                x = self.bridge(x)
            x = h + x
        return x


class ResidualNetwork(NN):

    def __init__(self, category_num, out_channels=(16, 32, 64), N=(18, 18, 18)):
        super(ResidualNetwork, self).__init__()
        # first conv
        self.conv1 = nn.Conv2d(3, out_channels[0], 3, 1, 1)
        in_channel = out_channels[0]
        # first block's stride is 1
        strides = [1] + [2] * (len(out_channels) - 1)
        # create resblock
        for i, out_channel, n, stride in six.moves.zip(six.moves.range(len(out_channels)), out_channels, N, strides):
            bridge = Bridge(in_channel, out_channel - in_channel, pool_flag=stride == 2)
            self['res_block{}'.format(i)] = ResBlock(in_channel, out_channel, n=n, stride_at_first_layer=stride, bridge=bridge)
            in_channel = out_channel
        self.bn_relu_conv = BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        # arguments
        self.out_channels = out_channels
        # name of model
        self.name = 'residual_network_{}_{}_{}'.format(category_num, out_channels, N)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        for i in six.moves.range(len(self.out_channels)):
            self['res_block{}'.format(i)].weight_initialization()
        self.bn_relu_conv.weight_initialization()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        for i in six.moves.range(len(self.out_channels)):
            h = self['res_block{}'.format(i)](h)
        h = self.bn_relu_conv(F.relu(h))
        h = self.global_average_pooling(h)
        return h
