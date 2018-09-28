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

    def forward(self, u, v):
        # u: Int[bs]
        # v: IntGPU[bs, 6]

        bag, offsets = self.model.get_bag(u.tolist(), self.u.weight.device)
        emb_u = self.u(bag, offsets)  # emb_u: Float[bs, d]

        emb_v = self.v(v)  # emb_v: Float[bs, 6, d]

        s = torch.einsum("ik,ijk->ij", (emb_u, emb_v))  # s: Float[bs, 6]
        return s

    @staticmethod
    def loss_fn(s):
        # s: FloatGPU[bs, 6]
        return -(F.logsigmoid(s[:, 0]).view(-1) + F.logsigmoid(-s[:, 1:]).sum(1)).mean()

    def get_input_matrix(self, n, bs, device=None):
        if n == -1:
            n = len(self.model.get_words())
        lst = []
        for i in range(0, n, bs):
            bag, offsets = self.model.get_bag(list(range(i, min(i + bs, n))), self.u.weight.device)
            if device is None:
                lst.append(self.u(bag, offsets))
            else:
                lst.append(self.u(bag, offsets).to(device))
        return torch.cat(lst, 0)
