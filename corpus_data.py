import gc

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _use_shared_memory

__all__ = ["CorpusData", "concat_collate", "BlockRandomSampler"]

BLOCK_SIZE = 1000000


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
    def __init__(self, corpus_path, dic_path, *, max_ws, n_ns, threshold, n_negatives=10000000, shuffle=False):
        self.corpus_path = corpus_path
        meta = torch.load(f"{corpus_path}.meta.pt")
        self.n_docs = meta["n_docs"]
        self.n_tokens = meta["n_tokens"]
        self.dic = torch.load(f"{dic_path}.pt")
        self.vocab_size = len(self.dic)
        self.negative_sampler = NegativeSampling(self.dic, n_negatives=n_negatives)
        self.max_ws = max_ws
        self.n_ns = n_ns
        self.shuffle = shuffle
        self.p_discard = get_discard_table(self.dic, self.n_tokens, threshold)

        self.corpus_block = None
        self.corpus_idx = -1

    def __getitem__(self, index):
        idx = (index // BLOCK_SIZE, index % BLOCK_SIZE)
        if idx[0] != self.corpus_idx:
            self.corpus_idx = idx[0]
            self.corpus_block = torch.load(f"{self.corpus_path}.{self.corpus_idx}.pt")
            gc.collect()
        doc = torch.IntTensor([w for w in self.corpus_block[idx[1]]
                               if w == self.vocab_size or np.random.rand() >= self.p_discard[w]])
        c, pos_u_b, pos_v_b, neg_v_b = 0, [], [], []
        for i in range(doc.shape[0]):
            pos_u = doc[i].item()
            ws = np.random.randint(1, self.max_ws + 1)
            for j in range(-ws, ws + 1):
                if j != 0 and 0 <= i + j < doc.shape[0]:
                    pos_v = doc[i + j].item()
                    neg_v = torch.LongTensor(self.n_ns)
                    for k in range(self.n_ns):
                        neg_v[k] = int(self.negative_sampler.sample(pos_v))
                    pos_u_b.append(pos_u)
                    pos_v_b.append(pos_v)
                    neg_v_b.append(neg_v)
                    c += 1
        pos_u_b = torch.LongTensor(pos_u_b).view(c, 1)
        pos_v_b = torch.LongTensor(pos_v_b).view(c, 1)
        if c > 0:
            neg_v_b = torch.stack(neg_v_b).view(c, self.n_ns)
        else:
            neg_v_b = torch.LongTensor([]).view(c, self.n_ns)
        if self.shuffle:
            perm = torch.randperm(c)
            return pos_u_b[perm], pos_v_b[perm], neg_v_b[perm]
        else:
            return pos_u_b, pos_v_b, neg_v_b

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


class BlockRandomSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        n_blocks = (len(self.data_source) + BLOCK_SIZE - 1) // BLOCK_SIZE
        lst = []
        for blk_id in torch.randperm(n_blocks):
            if blk_id == n_blocks - 1:
                blk_sz = len(self.data_source) - (n_blocks - 1) * BLOCK_SIZE
            else:
                blk_sz = BLOCK_SIZE
            lst += (torch.randperm(blk_sz) + blk_id * BLOCK_SIZE).tolist()
        return iter(lst)

    def __len__(self):
        return len(self.data_source)
