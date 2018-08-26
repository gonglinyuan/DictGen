import torch.nn as nn

__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    def __init__(self, emb_dim, *, n_layers, n_units, drop_prob, drop_prob_input, leaky, batch_norm):
        super(Discriminator, self).__init__()
        layers = []
        in_dim = emb_dim
        if drop_prob_input > 0:
            if batch_norm:
                layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Dropout(drop_prob_input))
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, n_units, bias=not batch_norm))
            in_dim = n_units
            if batch_norm:
                layers.append(nn.BatchNorm1d(in_dim))
            if leaky > 0:
                layers.append(nn.LeakyReLU(leaky))
            else:
                layers.append(nn.ReLU())
            if drop_prob > 0:
                layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(in_dim, 1, bias=not batch_norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1)
