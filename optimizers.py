import torch.optim as optim

__all__ = ["get_sgd", "get_rmsprop"]


class LinearDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_batch, last_epoch=-1):
        self.n_batch = n_batch
        self.factor = None
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.factor = 1 - self.last_epoch / self.n_batch
        return [self.factor * base_lr for base_lr in self.base_lrs]


class LinearLRWithApex(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_batch, apex, last_epoch=-1):
        self.n_batch = n_batch
        self.apex = apex
        self.factor = None
        super(LinearLRWithApex, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.factor = min(self.last_epoch / self.apex, (self.n_batch - self.last_epoch) / (self.n_batch - self.apex))
        return [self.factor * base_lr for base_lr in self.base_lrs]


def get_sgd(parameters, n_batch, *, lr, wd=0.0):
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    scheduler = LinearDecayLR(optimizer, n_batch)
    return optimizer, scheduler


def get_rmsprop(parameters, n_batch, *, lr, wd=0.0):
    optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=wd)
    scheduler = LinearDecayLR(optimizer, n_batch)
    return optimizer, scheduler


def get_adv(parameters, n_batch, *, lr, apex):
    optimizer = optim.SGD(parameters, lr=lr)
    scheduler = LinearLRWithApex(optimizer, n_batch, apex)
    return optimizer, scheduler
