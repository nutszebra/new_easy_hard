import argparse
from models import resnet, densenet
from utility.trainer_cifar10 import Cifar10Trainer
from utility.optimizers import MomentumSGD

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
args = parser.parse_args().__dict__
print('Args')
print('    {}'.format(args))
lr, momentum = args.pop('lr'), args.pop('momentum')
model, trainer = args.pop('model'), args.pop('trainer')

# define model
exec('model = {}'.format(model))
print('Model')
print('    name: {}'.format(model.name))
print('    parameters: {}'.format(model.count_parameters()))
# deine optimizer
optimizer = MomentumSGD(model, lr=lr, momentum=momentum, schedule=[100, 150], lr_decay=0.1)
optimizer.info()

args['model'], args['optimizer'] = model, optimizer
exec('main = {}(**args)'.format(trainer))
main.run()
