import sys

import numpy as np
import torch
from tqdm import trange, tqdm

from .wikicorpus_modified import WikiCorpus

if __name__ == "__main__":
    lang = sys.argv[1]
    wiki = WikiCorpus(f"{lang}wiki-20180801-pages-articles.xml.bz2")
    print(len(wiki))
    freqTable = np.zeros(len(wiki.dictionary) + 10, dtype=np.int64)
    for doc in tqdm(wiki):
        for id, freq in doc:
            freqTable[id] += freq
    sorted_ids = np.argsort(freqTable)[::-1]
    lst = []
    wd2id = {}
    for i in trange(200000):
        id = sorted_ids[i]
        wd2id[wiki.dictionary[id]] = i
        lst.append([wiki.dictionary[id], freqTable[id]])
    cor = []
    for doc in tqdm(wiki.get_texts(), total=len(wiki)):
        cor.append([wd2id.get(w, -1) for w in doc])
    torch.save(lst, f"{lang}_dictionary.pt")
    torch.save(cor, f"{lang}_wiki.pt")
