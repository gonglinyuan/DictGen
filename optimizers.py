import torch.optim as optim

__all__ = ["get"]


class LinearDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_batch, last_epoch=-1):
        self.n_batch = n_batch
        self.factor = None
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.factor = 1 - self.last_epoch / self.n_batch
        return [self.factor * base_lr for base_lr in self.base_lrs]


def get(parameters, n_batch, *, lr):
    optimizer = optim.SGD(parameters, lr=lr)
    scheduler = LinearDecayLR(optimizer, n_batch)
    return optimizer, scheduler
