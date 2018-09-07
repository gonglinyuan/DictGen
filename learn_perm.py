import argparse
import datetime
import os
from datetime import datetime

import fastText
import torch
import torch.nn as nn
import visdom
from tqdm import trange

import optimizers
from corpus_data import CorpusData
from skip_gram import SkipGram
from src.dictionary import Dictionary
from word_sampler import WordSampler

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")


class Permutation(nn.Module):
    def __init__(self, emb_dim, vocab_size, *, n_units):
        super(Permutation, self).__init__()
        layers = [nn.Linear(emb_dim, n_units), nn.ReLU(), nn.Linear(n_units, vocab_size), nn.Softmax(dim=1)]
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
        self.p_sample_top = params.p_sample_top
        # self.p_valid_top = params.p_valid_top
        self.emb_dim = params.emb_dim
        self.vocab_size_0, self.vocab_size_1 = corpus_data_0.vocab_size, corpus_data_1.vocab_size
        self.perm_optimizer, self.perm_scheduler = optimizers.get_sgd_adapt(self.perm.parameters(),
                                                                            lr=params.p_lr, wd=params.p_wd,
                                                                            momentum=params.p_momentum)

    def perm_step(self, *, fix_embedding=False):
        self.perm_optimizer.zero_grad()
        batch = [torch.LongTensor([self.sampler[id].sample() for _ in range(self.p_bs)]).view(self.p_bs, 1).to(GPU)
                 for id in [0, 1]]
        if fix_embedding:
            with torch.no_grad():
                x = [self.skip_gram[id].u(batch[id]).view(self.p_bs, -1) for id in [0, 1]]
        else:
            x = [self.skip_gram[id].u(batch[id]).view(self.p_bs, -1) for id in [0, 1]]
        x[0] = self.perm(x[0])
        if fix_embedding:
            with torch.no_grad():
                x[0] = torch.mm(x[0], self.skip_gram[1].u.weight[:self.p_sample_top])
        else:
            x[0] = torch.mm(x[0], self.skip_gram[1].u.weight[:self.p_sample_top])
        x = [torch.einsum("ik,jk->ij", (x[id], x[id])) for id in [0, 1]]
        loss = torch.mean((x[0] - x[1]) ** 2)
        loss.backward()
        self.perm_optimizer.step()
        return loss.item()

    # def valid_step(self):
    #     with torch.no_grad():
    #         batch0 = torch.arange(self.p_valid_top).view(self.p_valid_top, 1).to(GPU)  # Long[p_valid_top, 1]
    #         x0 = self.skip_gram[0].u(batch0).view(self.p_valid_top, -1)  # Float[p_valid_top, emb_dim]
    #         batch1 = self.perm(x0)  # Float[p_valid_top, p_sample_top]
    #         batch1 = torch.argmax(batch1, dim=1).view(self.p_valid_top, 1)  # Long[p_valid_top, 1]
    #         x1 = self.skip_gram[1].v(batch1).view(self.p_valid_top, -1)  # Float[p_valid_top, emb_dim]
    #         x0 = torch.einsum("ik,jk->ij", (x0, x0))
    #         x1 = torch.einsum("ik,jk->ij", (x1, x1))
    #         loss = torch.mean((x0 - x1) ** 2)
    #     return loss.item()

    def output(self):
        lst = []
        with torch.no_grad():
            for i in range(0, self.vocab_size_0, self.p_bs):
                batch = torch.arange(i, min(i + self.p_bs, self.vocab_size_0)).view(-1, 1).to(GPU)
                x = self.skip_gram[0].u(batch).view(-1, self.emb_dim)
                y = self.perm(x).topk(10, dim=1, largest=True, sorted=True)  # Long[p_bs, 10]
                lst.append(y)
        return torch.cat(lst)  # Long[vocab_size, 10]

    def scheduler_step(self, metric):
        self.perm_scheduler.step(metric)


def read_bin_embeddings(emb_path, emb_dim, dic):
    vocab_size = len(dic)
    u = torch.empty(vocab_size + 1, emb_dim, dtype=torch.float)
    v = torch.empty(vocab_size + 1, emb_dim, dtype=torch.float).normal_(mean=0, std=emb_dim ** (-0.25))
    model = fastText.load_model(emb_path)
    out_matrix = model.get_output_matrix()
    for i in range(vocab_size):
        u[i, :] = torch.from_numpy(model.get_word_vector(dic[i][0]))
        j = model.get_word_id(dic[i][0])
        v[i, :] = torch.from_numpy(out_matrix[j, :])
    return u, v


def normalize_embeddings(emb, types, mean=None):
    eps = 1e-3
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            return emb - mean.expand_as(emb)
        elif t == 'renorm':
            return emb / (emb.norm(2, 1, keepdim=True) + eps).expand_as(emb)
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return emb


def convert_dic(dic, lang):
    id2word, word2id = {}, {}
    for i in range(len(dic)):
        id2word[i] = dic[i][0]
        word2id[dic[i][0]] = i
    return Dictionary(id2word, word2id, lang)


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description="permutation learning")

    # Paths
    parser.add_argument("corpus_path_0", type=str, help="path of corpus 0")
    parser.add_argument("corpus_path_1", type=str, help="path of corpus 1")
    parser.add_argument("dic_path_0", type=str, help="path of dictionary 0")
    parser.add_argument("dic_path_1", type=str, help="path of dictionary 1")
    parser.add_argument("emb_path_0", type=str, help="path of embedding 0")
    parser.add_argument("emb_path_1", type=str, help="path of embedding 1")
    parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
    parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
    parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")

    # Global settings
    parser.add_argument("--src_lang", type=str, help="language of embedding 0")
    parser.add_argument("--tgt_lang", type=str, help="language of embedding 1")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--n_steps", type=int, help="number of steps per epoch")
    parser.add_argument("--epoch_tune_emb", type=int, help="the epoch to start tuning embeddings")
    parser.add_argument("--normalize_pre", type=str, default="", help="how to normalize the embedding before training")

    # Skip-gram settings
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--threshold", type=float, default=1e-4, help="sampling threshold")

    # Permutation learning settings
    parser.add_argument("--p_bs", type=int, help="batch size of permutation learning")
    parser.add_argument("--p_lr", type=float, help="learning rate of permutation learning")
    parser.add_argument("--p_momentum", type=float, help="momentum of permutation learning")
    parser.add_argument("--p_wd", type=float, help="weight decay of permutation learning")
    parser.add_argument("--p_n_units", type=int, help="number of hidden units in permutation model")
    parser.add_argument("--p_sample_top", type=int, help="sample top n frequent words in permutation learning")
    parser.add_argument("--p_sample_factor", type=float, help="sample factor of permutation learning")
    # parser.add_argument("--p_valid_top", type=int, help="sample top n frequent words in validation")

    params = parser.parse_args()

    print(params)

    out_path = os.path.join(params.modelDir, params.out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    corpus_data_0 = CorpusData(os.path.join(params.dataDir, params.corpus_path_0),
                               os.path.join(params.dataDir, params.dic_path_0),
                               max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
    corpus_data_1 = CorpusData(os.path.join(params.dataDir, params.corpus_path_1),
                               os.path.join(params.dataDir, params.dic_path_1),
                               max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
    trainer = Trainer(corpus_data_0, corpus_data_1, params=params)
    emb0_u, emb0_v = read_bin_embeddings(os.path.join(params.dataDir, params.emb_path_0),
                                         params.emb_dim, corpus_data_0.dic)
    emb1_u, emb1_v = read_bin_embeddings(os.path.join(params.dataDir, params.emb_path_1),
                                         params.emb_dim, corpus_data_1.dic)
    trainer.skip_gram[0].u.weight.data.copy_(normalize_embeddings(emb0_u, params.normalize_pre))
    trainer.skip_gram[0].v.weight.data.copy_(normalize_embeddings(emb0_v, params.normalize_pre))
    trainer.skip_gram[1].u.weight.data.copy_(normalize_embeddings(emb1_u, params.normalize_pre))
    trainer.skip_gram[1].v.weight.data.copy_(normalize_embeddings(emb1_v, params.normalize_pre))
    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(out_path, "log.txt"), use_incoming_socket=False)
    out_freq = 500
    c, step, p_loss = 0, 0, 0.0
    v_loss, v_norm = 0.0, 0
    for epoch in trange(params.n_epochs):
        for _ in trange(params.n_steps):
            p_loss += trainer.perm_step(fix_embedding=epoch >= params.epoch_tune_emb)
            c += 1
            v_loss = v_loss * 0.999 + p_loss
            v_norm = v_norm * 0.999 + 1.0
            if c >= out_freq:
                vis.line(Y=torch.FloatTensor([p_loss / c]), X=torch.LongTensor([step]),
                         win="p_loss", env=params.out_path, opts={"title": "p_loss"}, update="append")
                c, p_loss = 0, 0.0
                step += 1
        print(f"epoch {epoch} loss is {v_loss / v_norm}")
        trainer.scheduler_step(v_loss)
        v_loss, v_norm = 0.0, 0
        dic0, dic1 = convert_dic(corpus_data_0.dic, params.src_lang), convert_dic(corpus_data_1.dic, params.tgt_lang)
        model_output = trainer.output()
        torch.save({"dic0": dic0, "dic1": dic1, "out": model_output}, os.path.join(out_path, f"out-epoch{epoch}.pth"))
    print(params)


if __name__ == '__main__':
    main()
