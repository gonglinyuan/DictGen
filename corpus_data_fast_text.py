import collections

import fastText
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _use_shared_memory

from word_sampler import WordSampler

__all__ = ["CorpusData", "concat_collate"]


def get_discard_table(dic, n_tokens, threshold):
    p_discard = np.zeros(len(dic))
    for i in range(len(dic)):
        p_discard[i] = dic[i][1] / n_tokens
    p_discard = threshold / p_discard
    p_discard = p_discard ** 0.5 + p_discard
    p_discard = 1.0 - p_discard
    p_discard = np.maximum(p_discard, 0.0, out=p_discard)
    return p_discard


class CorpusData(Dataset):
    def __init__(self, path_data, path_model, *, max_ws, n_ns, threshold, n_negatives=10000000):
        with open(path_data, mode="r", encoding="utf-8") as f:
            self.n_docs = sum(1 for _ in f)
        self.model = fastText.load_model(path_model)
        self.dic = list(zip(*self.model.get_words(include_freq=True)))
        self.n_tokens = sum(freq for _, freq in self.dic)
        self.file = open(path_data, mode="r", encoding="utf-8")
        self.p_discard = get_discard_table(self.dic, self.n_tokens, threshold)
        self.negative_sampler = WordSampler(self.dic, n_urns=n_negatives, alpha=0.5)
        self.max_ws = max_ws
        self.n_ns = n_ns

    def __getitem__(self, index):
        doc = self.file.readline()
        if not doc:
            self.file.seek(0)
            doc = self.file.readline()
        doc = self.model.get_line(doc.strip())[0]
        doc = [self.model.get_word_id(w) for w in doc]
        doc = [w for w in doc if w != -1 and np.random.rand() >= self.p_discard[w]]
        c, u_b, v_b = 0, [], []
        for i in range(len(doc)):
            u = doc[i]
            ws = np.random.randint(1, self.max_ws + 1)
            for j in range(-ws, ws + 1):
                if j != 0 and 0 <= i + j < len(doc):
                    v = torch.LongTensor(self.n_ns + 1)
                    v[0] = doc[i + j]
                    for k in range(1, self.n_ns + 1):
                        v[k] = int(self.negative_sampler.sample_neg(doc[i + j]))
                    u_b.append(u)
                    v_b.append(v)
                    c += 1
        u_b = torch.LongTensor(u_b)
        if c > 0:
            v_b = torch.stack(v_b).view(c, self.n_ns + 1)
        else:
            v_b = torch.LongTensor([]).view(c, self.n_ns + 1)
        return u_b, v_b  # u_b: Int[c] ; v_b: Int[c, 6]

    def __len__(self):
        return self.n_docs


def concat_collate(batch):
    result = []
    sz = None
    for samples in zip(*batch):
        tmp = torch.cat(samples, 0)
        sz = tmp.shape[0]
        result.append(tmp)
    perm = torch.randperm(sz)
    if _use_shared_memory:
        return [tmp[perm].share_memory_() for tmp in result]
    else:
        return [tmp[perm] for tmp in result]
