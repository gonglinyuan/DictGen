import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FastText"]


class FastText(nn.Module):
    def __init__(self, model):
        super(FastText, self).__init__()
        self.model = model
        self.emb_dim = model.get_dimension()
        data_u = torch.from_numpy(model.get_input_matrix())
        data_v = torch.from_numpy(model.get_output_matrix())
        self.u = nn.EmbeddingBag(data_u.shape[0], self.emb_dim, mode="mean", sparse=True)
        self.u.weight.data.copy_(data_u)
        self.v = nn.Embedding.from_pretrained(data_v, sparse=True)

    def forward(self, pos_u, pos_v, neg_v):
        # pos_u: String[bs]
        # pos_v: Int[bs]
        # neg_v: Int[bs, 5]
        bs = len(pos_u)

        bag, offsets = [], []
        for w in pos_u:
            offsets.append(len(bag))
            bag += self.model.get_subwords(w)[1]
        emb_u = self.u(torch.LongTensor(bag), offsets)  # emb_u: Float[bs, d]

        v = torch.LongTensor(bs, 6)
        for i in range(bs):
            v[i, 0] = pos_v[i]
            for j in range(5):
                v[i, j + 1] = neg_v[i, j]
        emb_v = self.v(v)  # emb_v: Float[bs, 6, d]

        s = torch.einsum("ik,ijk->ij", (emb_u, emb_v))  # s: Float[bs, 6]
        return s

    @staticmethod
    def loss_fn(s):
        # pos_s: Float[bs, 1]
        # neg_s: Float[bs, 5]
        return -(F.logsigmoid(s[:, 0]).view(-1) + F.logsigmoid(-s[:, 1:]).sum(1)).mean()
