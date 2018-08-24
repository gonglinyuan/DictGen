import numpy as np


class WordSampler:
    def __init__(self, dic, *, n_urns, alpha):
        self.n_urns = n_urns
        self.urn = np.zeros(self.n_urns, dtype=np.int32)
        j = 0
        z = 0.0
        for i in range(len(dic)):
            z += dic[i][1] ** alpha
        for i in range(len(dic)):
            c = dic[i][1] ** alpha
            for _ in range(int(c * self.n_urns / z)):
                self.urn[j] = i
                j += 1
        np.random.shuffle(self.urn)
        self.p = 0

    def sample(self):
        s = self.urn[self.p]
        self.p = (self.p + 1) % self.n_urns
        return s

    def sample_neg(self, pos):
        neg = self.urn[self.p]
        self.p = (self.p + 1) % self.n_urns
        while neg == pos:
            neg = self.urn[self.p]
            self.p = (self.p + 1) % self.n_urns
        return neg