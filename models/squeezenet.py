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


class FireModule(NN):

    def __init__(self, in_size, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.bn_relu_conv_s_1x1 = BN_ReLU_Conv(in_size, s1x1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        self.bn_relu_conv_e_1x1 = BN_ReLU_Conv(s1x1, e1x1, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        self.bn_relu_conv_e_3x3 = BN_ReLU_Conv(s1x1, e3x3, filter_size=(3, 3), stride=(1, 1), pad=(1, 1))

    def weight_initialization(self):
        self.bn_relu_conv_s_1x1.weight_initialization()
        self.bn_relu_conv_e_1x1.weight_initialization()
        self.bn_relu_conv_e_3x3.weight_initialization()

    def forward(self, x):
        h = self.bn_relu_conv_s_1x1(x)
        h1 = self.bn_relu_conv_e_1x1(h)
        h2 = self.bn_relu_conv_e_3x3(h)
        return torch.cat((h1, h2), 1)


class SqueezeNet(NN):

    def __init__(self, category_num):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, (7, 7), (2, 2), (2, 2))
        # fire module(in_size, s1x1, e1x1, e3x3)
        self['fire2'] = FireModule(96, 16, 64, 64)
        self['fire3'] = FireModule(128, 16, 64, 64)
        self['fire4'] = FireModule(128, 32, 128, 128)
        self['fire5'] = FireModule(256, 32, 128, 128)
        self['fire6'] = FireModule(256, 48, 192, 192)
        self['fire7'] = FireModule(384, 48, 192, 192)
        self['fire8'] = FireModule(384, 64, 256, 256)
        self['fire9'] = FireModule(512, 64, 256, 256)
        self.bn_relu_conv10 = BN_ReLU_Conv(512, category_num, (1, 1), (1, 1), (0, 0))
        self.drop = nn.Dropout(p=0.5)
        self.name = 'squeeze_res_net_{}'.format(category_num)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        # *****fire modules*****
        for i in six.moves.range(2, 10):
            self['fire{}'.format(i)].weight_initialization()
        self.bn_relu_conv10.weight_initialization()

    def forward(self, x, train=True):
        h = self.conv1(x)
        h = F.max_pool2d(h, (3, 3), (2, 2), (1, 1))
        h = self.fire2(h)
        h = self.fire3(h) + h
        h = self.fire4(h)
        h = F.max_pool2d(h, (3, 3), (2, 2), (1, 1))
        h = self.fire5(h) + h
        h = self.fire6(h)
        h = self.fire7(h) + h
        h = self.fire8(h)
        h = F.max_pool2d(h, (3, 3), (2, 2), (1, 1))
        h = self.drop(self.fire9(h))
        h = self.bn_relu_conv10(h)
        h = self.global_average_pooling(h)
        return h
