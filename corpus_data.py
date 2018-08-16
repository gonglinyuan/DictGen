import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import _use_shared_memory

__all__ = ["CorpusData", "concat_collate"]


class NegativeSampling:
    def __init__(self, dic, *, n_negatives):
        self.n_negatives = n_negatives
        self.negatives = np.zeros(self.n_negatives, dtype=np.int32)
        j = 0
        z = 0.0
        for i in range(len(dic)):
            z += dic[i][1] ** 0.5
        for i in range(len(dic)):
            c = dic[i][1] ** 0.5
            for _ in range(int(c * self.n_negatives / z)):
                self.negatives[j] = i
                j += 1
        np.random.shuffle(self.negatives)
        self.neg_pos = 0

    def sample(self, pos):
        neg = self.negatives[self.neg_pos]
        self.neg_pos = (self.neg_pos + 1) % self.n_negatives
        while neg == pos:
            neg = self.negatives[self.neg_pos]
            self.neg_pos = (self.neg_pos + 1) % self.n_negatives
        return neg


class CorpusData(Dataset):
    def __init__(self, corpus_path, dic_path, *, max_ws, n_ns, n_negatives=10000000, shuffle=False):
        self.corpus = torch.load(corpus_path)
        self.dic = torch.load(dic_path)
        self.vocab_size = len(self.dic)
        self.negative_sampler = NegativeSampling(self.dic, n_negatives=n_negatives)
        self.max_ws = max_ws
        self.n_ns = n_ns
        self.shuffle = shuffle
        for doc in self.corpus:
            for i in range(len(doc)):
                if doc[i] == -1:
                    doc[i] = self.vocab_size

    def __getitem__(self, index):
        doc = self.corpus[index]
        c, pos_u_b, pos_v_b, neg_v_b = 0, [], [], []
        for i in range(len(doc)):
            pos_u = doc[i]
            ws = np.random.randint(1, self.max_ws + 1)
            for j in range(-ws, ws + 1):
                if j != 0 and 0 <= i + j < len(doc):
                    pos_v = doc[i + j]
                    neg_v = []
                    for _ in range(self.n_ns):
                        neg_v.append(self.negative_sampler.sample(pos_v))
                    pos_u_b.append(pos_u)
                    pos_v_b.append(pos_v)
                    neg_v_b.append(neg_v)
                    c += 1
        pos_u_b = torch.LongTensor(pos_u_b).view(c, 1)
        pos_v_b = torch.LongTensor(pos_v_b).view(c, 1)
        neg_v_b = torch.LongTensor(neg_v_b).view(c, self.n_ns)
        if self.shuffle:
            perm = torch.randperm(c)
            return pos_u_b[perm], pos_v_b[perm], neg_v_b[perm]
        else:
            return pos_u_b, pos_v_b, neg_v_b

    def __len__(self):
        return len(self.corpus)


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
