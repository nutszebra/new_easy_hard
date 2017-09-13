import argparse
from models import resnet, densenet, squeezenet, alexnet, vgg_a, resnext, shakeshake
from utility.trainer_cifar10 import Cifar10Trainer
from utility.trainer_cifar100 import Cifar100Trainer
from utility.optimizers import MomentumSGD
import utility.transformers as transformers
from utility import utility

parser = argparse.ArgumentParser(description='PyTorch cifar10 Example')
parser.add_argument('--gpu', type=int, default=-1, metavar='N',
                    help='-1 means cpu, otherwise gpu id')
parser.add_argument('--save_path', type=str, default='./log', metavar='N',
                    help='log and model will be saved here')
parser.add_argument('--load_model', default=None, metavar='N',
                    help='pretrained model')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='epochs start from this number')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--model', type=str, default='resnet.ResidualNetwork(10, out_channels=(16, 32, 64), N=(18, 18, 18))', metavar='M',
                    help='model definition here')
parser.add_argument('--trainer', type=str, default='Cifar10Trainer', metavar='M',
                    help='model definition here')
parser.add_argument('--train_transform', type=str, default=None, metavar='M',
                    help='train transform')
parser.add_argument('--test_transform', type=str, default=None, metavar='M',
                    help='train transform')
args = parser.parse_args().__dict__
print('Args')
print('    {}'.format(args))
lr, momentum = args.pop('lr'), args.pop('momentum')
model_name, trainer_name = args.pop('model'), args.pop('trainer')
# deine optimizer

exec('{}={}'.format("args['train_transform']", args['train_transform']))
exec('{}={}'.format("args['test_transform']", args['test_transform']))

for i in utility.create_progressbar(args['epochs'], desc='hard', start=args['start_epoch']):
    # define model
    exec('model = {}'.format(model_name))
    print('Model')
    print('    name: {}'.format(model.name))
    print('    parameters: {}'.format(model.count_parameters()))
    # deine fake optimizer
    optimizer = MomentumSGD(model, lr=lr, momentum=momentum, schedule=[100, 150], lr_decay=0.1)
    args['model'], args['optimizer'] = model, optimizer
    exec('main = {}(**args)'.format(trainer_name))
    main.train_one_epoch()
    results = main.test_one_epoch(keep=True)
