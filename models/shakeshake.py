import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from .prototype import NN
from .mul import mul


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


class ReLU_Conv_BN_ReLU_Conv_BN(NN):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(ReLU_Conv_BN_ReLU_Conv_BN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, filter_size[0], stride[0], pad[0])
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, filter_size[1], stride[1], pad[1])
        self.bn2 = nn.BatchNorm2d(out_channel)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.conv2.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv2))
        self.conv2.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv2, constant=0))

    def forward(self, x):
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class Double(NN):

    def __init__(self, in_channel, out_channel):
        out_channel1 = int(out_channel) / 2
        out_channel2 = out_channel - out_channel1
        super(Double, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, int(out_channel1), 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channel, int(out_channel2), 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channel)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.conv2.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv2))
        self.conv2.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv2, constant=0))

    def forward(self, x):
        x1 = self.conv1(F.avg_pool2d(x, 1, 2, 0))
        x2 = self.conv2(F.avg_pool2d(F.pad(x[:, :, 1:, 1:], (1, 0, 1, 0)), 1, 2, 0))
        return self.bn(torch.cat((x1, x2), 1))


class ResBlock(NN):

    def __init__(self, in_channel, out_channel, branch_num=2, filter_size=(3, 3), stride=(1, 1), pad=(1, 1), identity=NN()):
        super(ResBlock, self).__init__()
        for i in six.moves.range(branch_num):
            self['branch{}'.format(i)] = ReLU_Conv_BN_ReLU_Conv_BN(in_channel, out_channel, filter_size, stride, pad)
            self.branch_num = branch_num
        self.identity = identity

    def weight_initialization(self):
        for i in six.moves.range(self.branch_num):
            self['branch{}'.format(i)].weight_initialization()
        self.identity.weight_initialization()

    def forward(self, x):
        branches = []
        for i in six.moves.range(self.branch_num):
            branches.append(self['branch{}'.format(i)](x))
        x = self.identity(x)
        return mul(*branches, train=not x.volatile) + x


class ShakeShake(NN):

    def __init__(self, category_num, out_channels=(64, 128, 256), N=(4, 4, 4), branch_num=2):
        super(ShakeShake, self).__init__()
        # conv
        self.conv1 = nn.Conv2d(3, out_channels[0], 3, 1, 1)
        in_channel = out_channels[0]
        strides = [[(1, 1) for i in six.moves.range(N[ii])] for ii in six.moves.range(len(out_channels))]
        identities = [[NN() for i in six.moves.range(N[ii])] for ii in six.moves.range(len(out_channels))]
        strides[1][0], identities[1][0] = (2, 1), Double(64, 128)
        strides[2][0], identities[2][0] = (2, 1), Double(128, 256)
        for i in six.moves.range(len(out_channels)):
            for n in six.moves.range(N[i]):
                self['res_block{}_{}'.format(i, n)] = ResBlock(in_channel, out_channels[i], branch_num, (3, 3), strides[i][n], (1, 1), identities[i][n])
                in_channel = out_channels[i]
        self.linear = BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))
        self.out_channels, self.N = out_channels, N
        self.name = 'shake_shake_{}_{}_{}_{}'.format(category_num, out_channels, N, branch_num)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        for i in six.moves.range(len(self.out_channels)):
            for n in six.moves.range(self.N[i]):
                self['res_block{}_{}'.format(i, n)].weight_initialization()
        self.linear.weight_initialization()

    def forward(self, x):
        h = F.relu(self.conv1(x))
        for i in six.moves.range(len(self.out_channels)):
            for n in six.moves.range(self.N[i]):
                h = self['res_block{}_{}'.format(i, n)](h)
                print(h.data.shape)
        h = self.linear(h)
        h = self.global_average_pooling(h)
        return h
