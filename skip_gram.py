import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SkipGram"]


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.u = nn.Embedding(vocab_size, emb_dim, sparse=True)
        self.v = nn.Embedding(vocab_size, emb_dim, sparse=True)
        nn.init.normal_(self.u.weight, mean=0, std=emb_dim ** (-0.25))
        nn.init.normal_(self.v.weight, mean=0, std=emb_dim ** (-0.25))

    def forward(self, pos_u, pos_v, neg_v):
        # pos_u: Int[bs, 1]
        # pos_v: Int[bs, 1]
        # neg_v: Int[bs, 5]
        bs = pos_u.shape[0]
        emb_pos_u = self.u(pos_u)  # emb_pos_u: Int[bs, 1, d]
        emb_pos_v = self.v(pos_v)  # emb_pos_v: Int[bs, 1, d]
        emb_neg_v = self.v(neg_v)  # emb_neg_v: Int[bs, 5, d]
        pos_s = torch.bmm(emb_pos_v, emb_pos_u.view(bs, self.emb_dim, 1)).view(bs, -1)  # pos_s: Float[bs, 1]
        neg_s = torch.bmm(emb_neg_v, emb_pos_u.view(bs, self.emb_dim, 1)).view(bs, -1)  # neg_s: Float[bs, 5]
        return pos_s, neg_s

    @staticmethod
    def loss_fn(pos_s, neg_s):
        # pos_s: Float[bs, 1]
        # neg_s: Float[bs, 5]
        return -(F.logsigmoid(pos_s).view(-1) + F.logsigmoid(-neg_s).sum(1)).mean()
