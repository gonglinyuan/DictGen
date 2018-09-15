import torch
import torch.nn as nn

__all__ = ["Discriminator"]


def _clip_elementwise(module):
    m = 1.0 / (module.weight.size(1) ** 0.5)
    module.weight.data.clamp_(-m, m)


def _clip_spectral(module):
    with torch.no_grad():
        u, s, vt = torch.svd(module.weight.data)
        s.clamp_(0.0, 1.0)
        module.weight.data.copy_(torch.einsum("ik,k,kj->ij", (u, s, vt)))


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

    def clip_weights(self, mode="elementwise"):
        for module in self.layers:
            if isinstance(module, nn.Linear):
                if mode == "elementwise":
                    _clip_elementwise(module)
                elif mode == "spectral":
                    _clip_spectral(module)
                elif mode == "none":
                    pass
                else:
                    raise Exception(f"clip mode {mode} not found")
