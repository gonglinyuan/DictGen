import torch.nn as nn

__all__ = ["Discriminator"]


class Discriminator(nn.Module):
    def __init__(self, emb_dim, *, n_layers, n_units, drop_prob, drop_prob_input, leaky):
        super(Discriminator, self).__init__()
        layers = []
        if drop_prob_input > 0:
            layers.append(nn.Dropout(drop_prob_input))
        in_dim = emb_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, n_units))
            in_dim = n_units
            if leaky > 0:
                layers.append(nn.LeakyReLU(leaky))
            else:
                layers.append(nn.ReLU())
            if drop_prob > 0:
                layers.append(nn.Dropout(drop_prob))
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1)
