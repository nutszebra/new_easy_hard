import torch
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable


class Mul(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        p1 = input.new().resize_as_(input)
        p2 = input.new().resize_as_(input)
        p2.fill_(1.0)
        for i in range(p1.shape[0]):
            p1[i, :] = torch.rand(1)[0]
        p2 = p2 - p1
        return p1, p2

    def _make_test(input):
        p = input.new().resize_as_(input)
        p.fill_(0.5)
        return p, p

    @staticmethod
    def forward(ctx, input1, input2, train=False):
        if train:
            p1, p2 = Mul._make_test(input1)
        else:
            p1, p2 = Mul._make_noise(input1)
        return p1 * input2 + p2 * input2

    @staticmethod
    def backward(ctx, grad_output):
        p1, p2 = Mul._make_noise(grad_output.data)
        return Variable(grad_output.data * p1), Variable(grad_output.data * p2), None


def mul(x1, x2, train=False):
    return Mul.apply(x1, x2, train)


if __name__ == '__main__':
    x1 = Variable(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]), volatile=False, requires_grad=True)
    x2 = Variable(torch.FloatTensor([[2, 3, 4], [5, 6, 7]]), volatile=False, requires_grad=True)
    t = Variable(torch.FloatTensor([[1, 1, 1], [1, 1, 1]]), volatile=False, requires_grad=True)
    x = Mul.apply(x1, x2)
    loss = torch.sum(x - t)
    loss.backward()
