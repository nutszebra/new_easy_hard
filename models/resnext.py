import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prototype import NN


class Linear(NN):

    def __init__(self, in_channel, out_channel):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel)

    def weight_initialization(self):
        self.fc.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.fc))
        self.fc.bias.data = torch.FloatTensor(NN.bias_initialization(self.fc, constant=0))

    def forward(self, x):
        return self.fc(x)


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


class Conv_BN(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_channel)

    def weight_initialization(self):
        self.conv.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv))
        self.conv.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv, constant=0))

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv_BN_ReLU(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_channel)

    def weight_initialization(self):
        self.conv.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv))
        self.conv.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv, constant=0))

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Group_Conv(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), groups=32):
        super(Group_Conv, self).__init__()
        self.group_conv = nn.Conv2d(in_channel, out_channel, filter_size, stride, pad, groups=groups)

    def weight_initialization(self):
        self.group_conv.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.group_conv))
        self.group_conv.bias.data = torch.FloatTensor(NN.bias_initialization(self.group_conv, constant=0))

    def forward(self, x):
        return self.group_conv(x)


class ResNextBlock(NN):

    def __init__(self, in_channel=64, out_channels=(256, 256, 256), filters=(1, 3, 1), strides=(1, 1, 1), pads=(0, 1, 0), C=32, skip_connection=NN()):
        super(ResNextBlock, self).__init__()
        self.conv_bn_relu_1 = Conv_BN_ReLU(in_channel, out_channels[0], filters[0], strides[0], pads[0])
        self.group_conv_2 = Group_Conv(out_channels[0], out_channels[1], filters[1], strides[1], pads[1], groups=C)
        self.bn_relu_conv_3 = BN_ReLU_Conv(out_channels[1], out_channels[2], filters[2], strides[2], pads[2])
        self.bn4 = nn.BatchNorm2d(out_channels[2])
        self.skip_connection = skip_connection

    def weight_initialization(self):
        self.conv_bn_relu_1.weight_initialization()
        self.group_conv_2.weight_initialization()
        self.bn_relu_conv_3.weight_initialization()

    def __call__(self, x):
        h = self.conv_bn_relu_1(x)
        h = self.group_conv_2(h)
        h = self.bn_relu_conv_3(h)
        h = self.bn4(h)
        x = h + self.skip_connection(x)
        return F.relu(x)


class ResNext(NN):

    def __init__(self, category_num, block_num=(3, 3, 3), C=8, d=64, multiplier=4):
        super(ResNext, self).__init__()
        weight_init_queue = []
        # conv
        self.conv_bn_relu = Conv_BN_ReLU(3, 64, 7, 2, 3)
        weight_init_queue.append(self.conv_bn_relu)
        out_channels = [(C * d * i, C * d * i, d * i * multiplier) for i in [2 ** x for x in six.moves.range(len(block_num))]]
        in_channel = 64
        for i, n in enumerate(block_num):
            for ii in six.moves.range(n):

                if i >= 1 and ii == 0:
                    strides = (1, 2, 1)
                    skip_connection = Conv_BN(in_channel, out_channels[i][-1], 1, 2, 0)
                elif i == 0 and ii == 0:
                    strides = (1, 1, 1)
                    skip_connection = Conv_BN(in_channel, out_channels[i][-1], 1, 1, 0)
                else:
                    strides = (1, 1, 1)
                    skip_connection = NN()
                self['resnext_block_{}_{}'.format(i, ii)] = ResNextBlock(in_channel, out_channels[i], (1, 3, 1), strides, (0, 1, 0), C, skip_connection=skip_connection)
                weight_init_queue.append(self['resnext_block_{}_{}'.format(i, ii)])
                in_channel = out_channels[i][-1]
        self.linear = Linear(in_channel, category_num)
        weight_init_queue.append(self.linear)
        self.weight_init_queue = weight_init_queue
        self.C = C
        self.block_num = block_num
        self.name = 'ResNext_{}_{}'.format(category_num, C)

    def weight_initialization(self):
        [link.weight_initialization() for link in self.weight_init_queue]

    def forward(self, x):
        h = self.conv_bn_relu(x)
        h = F.max_pool2d(h, (3, 3), (2, 2), (1, 1))
        for i, n in enumerate(self.block_num):
            for ii in six.moves.range(n):
                h = self['resnext_block_{}_{}'.format(i, ii)](h)
        h = self.global_average_pooling(h)
        return self.linear(h)
