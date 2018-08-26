import sys
from array import array

import numpy as np
import torch
from tqdm import trange, tqdm

from wikicorpus_modified import WikiCorpus

VOCAB_SIZE = 400000
BLOCK_SIZE = 1000000

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
    for i in trange(VOCAB_SIZE):
        id = sorted_ids[i]
        wd2id[wiki.dictionary[id]] = i
        lst.append([wiki.dictionary[id], freqTable[id]])
    cor, tot, blk_cnt, n_tokens = [], len(wiki), 0, 0
    for i, doc in tqdm(enumerate(wiki.get_texts()), total=tot):
        cor.append(array("i", [wd2id.get(w, VOCAB_SIZE) for w in doc]))
        n_tokens += len(doc)
        if (i + 1) % BLOCK_SIZE == 0 or i + 1 == tot:
            torch.save(cor, f"{lang}_cor.{blk_cnt}.pt")
            cor = []
            blk_cnt += 1
    torch.save(lst, f"{lang}_dic.pt")
    torch.save({
        "n_docs": tot,
        "n_tokens": n_tokens
    }, f"{lang}_cor.meta.pt")
