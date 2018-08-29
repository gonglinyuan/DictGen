import torch.optim as optim

__all__ = ["get_sgd_linear", "get_rmsprop_linear", "get_sgd_exp", "get_sgd_adapt"]


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


def get_sgd_linear(parameters, n_batch, *, lr, wd=0.0):
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    scheduler = LinearDecayLR(optimizer, n_batch)
    return optimizer, scheduler


def get_rmsprop_linear(parameters, n_batch, *, lr, wd=0.0):
    optimizer = optim.RMSprop(parameters, lr=lr, weight_decay=wd)
    scheduler = LinearDecayLR(optimizer, n_batch)
    return optimizer, scheduler


def get_sgd_exp(parameters, factor, *, lr, wd=0.0):
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, factor)
    return optimizer, scheduler


def get_sgd_cosine(parameters, n_batch, *, lr, wd=0.0):
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batch)
    return optimizer, scheduler


def get_sgd_adapt(parameters, *, lr, wd=0.0):
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=True)
    return optimizer, scheduler
