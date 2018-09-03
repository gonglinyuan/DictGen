import sys
from array import array

import torch
from tqdm import trange, tqdm

VOCAB_SIZE = 400000
BLOCK_SIZE = 1000000

if __name__ == "__main__":
    lang = sys.argv[1]
    path = sys.argv[2]

    word2freq = {}
    tot = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            tot += 1
            for w in line.split():
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

    dic = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)[:VOCAB_SIZE]
    del word2freq
    word2id = {}
    for i in trange(VOCAB_SIZE):
        word2id[dic[i][0]] = i
    cor, n_tokens, blk_cnt = [], 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), total=tot):
            para = line.split()
            cor.append(array("i", [word2id.get(w, VOCAB_SIZE) for w in para]))
            n_tokens += len(para)
            if (i + 1) % BLOCK_SIZE == 0 or i + 1 == tot:
                torch.save(cor, f"{lang}_cor.{blk_cnt}.pt")
                cor = []
                blk_cnt += 1
    torch.save(dic, f"{lang}_dic.pt")
    torch.save({
        "n_docs": tot,
        "n_tokens": n_tokens
    }, f"{lang}_cor.meta.pt")
