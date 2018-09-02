import torch
import torch.nn as nn
from skip_gram import SkipGram
from word_sampler import WordSampler

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")

class Permutation(nn.Module):
    def __init__(self, emb_dim, vocab_size, *, n_units):
        super(Permutation, self).__init__()
        layers = [nn.Linear(emb_dim, n_units), nn.ReLU(), nn.Linear(n_units, vocab_size), nn.Softmax()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Trainer:
    def __init__(self, corpus_data_0, corpus_data_1, *, params, n_samples=10000000):
        self.skip_gram = [SkipGram(corpus_data_0.vocab_size + 1, params.emb_dim).to(GPU),
                          SkipGram(corpus_data_1.vocab_size + 1, params.emb_dim).to(GPU)]
        self.perm = Permutation(params.emb_dim, params.p_sample_top, n_units=params.p_n_units).to(GPU)
        self.sampler = [
            WordSampler(corpus_data_0.dic, n_urns=n_samples, alpha=params.p_sample_factor, top=params.p_sample_top),
            WordSampler(corpus_data_1.dic, n_urns=n_samples, alpha=params.p_sample_factor, top=params.p_sample_top)]
        self.p_bs = params.p_bs

    def get_batch(self, *, fix_embedding=False):
        batch = [torch.LongTensor([self.sampler[id].sample() for _ in range(self.p_bs)]).view(self.p_bs, 1).to(GPU)
                 for id in [0, 1]]
        if fix_embedding:
            with torch.no_grad():
                x = [self.skip_gram[id].u(batch[id]).view(self.p_bs, -1) for id in [0, 1]]
        else:
            x = [self.skip_gram[id].u(batch[id]).view(self.p_bs, -1) for id in [0, 1]]
        x[1] = self.perm(x[1])
        if fix_embedding:
            with torch.no_grad():
                x[1] = torch.mm(x[1], self.skip_gram[0].u.weight)
        else:
            x[1] = torch.mm(x[1], self.skip_gram[0].u.weight)
        x = [torch.einsum("ik,jk->ij", (x[id], x[id])) for id in [0, 1]]
        torch.sum((x[0] - x[1]) ** 2)
