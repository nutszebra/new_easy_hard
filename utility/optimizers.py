import torch.optim as optim
import utility


class MomentumSGD(object):

    def __init__(self, model, lr, momentum, schedule=[10, 20], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        utility.write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                utility.write('    {}: {}'.format(key, self.__dict__[key]))
