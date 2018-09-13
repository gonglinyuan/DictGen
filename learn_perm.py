import argparse
import datetime
import os
from datetime import datetime

import fastText
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, emb_dim, vocab_size, *, n_units, batch_norm):
        super(Permutation, self).__init__()
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm1d(emb_dim))
        layers.append(nn.Linear(emb_dim, n_units, bias=not batch_norm))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_units))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_units, vocab_size, bias=not batch_norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        return -torch.sum(F.softmax(x, dim=1) * F.log_softmax(x, dim=1), dim=1).mean()


def correlation_distribution(x):
    u, s, _ = torch.svd(x)
    mx = torch.einsum("ik,k,jk->ij", (u, s, u))
    mx, _ = torch.sort(mx, dim=1, descending=True)
    return mx


def csls_knn(x, z, k=10):
    x /= x.norm(p=2, dim=1, keepdim=True).expand_as(x)
    z /= z.norm(p=2, dim=1, keepdim=True).expand_as(z)
    sim = torch.einsum("ik,jk->ij", (x, z))
    sx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)[0].mean(dim=1, keepdim=True).expand_as(sim)
    sz = torch.topk(sim, k=k, dim=0, largest=True, sorted=False)[0].mean(dim=0, keepdim=True).expand_as(sim)
    sim = sim * 2 - sx - sz
    return torch.argmax(sim, dim=1)


class Trainer:
    def __init__(self, corpus_data_0, corpus_data_1, *, params, n_samples=10000000):
        self.skip_gram = [SkipGram(corpus_data_0.vocab_size + 1, params.emb_dim).to(GPU),
                          SkipGram(corpus_data_1.vocab_size + 1, params.emb_dim).to(GPU)]
        self.perm = Permutation(params.emb_dim, params.p_sample_top, n_units=params.p_n_units,
                                batch_norm=params.p_bn).to(GPU)
        self.sampler = [
            WordSampler(corpus_data_0.dic, n_urns=n_samples, alpha=params.p_sample_factor, top=params.p_sample_top),
            WordSampler(corpus_data_1.dic, n_urns=n_samples, alpha=params.p_sample_factor, top=params.p_sample_top)]
        self.p_bs = params.p_bs
        self.i_bs = params.i_bs
        self.p_sample_top = params.p_sample_top
        self.emb_dim = params.emb_dim
        self.vocab_size_0, self.vocab_size_1 = corpus_data_0.vocab_size, corpus_data_1.vocab_size
        self.perm_optimizer, self.perm_scheduler = optimizers.get_sgd_adapt(self.perm.parameters(), lr=params.p_lr,
                                                                            mode="min", wd=params.p_wd,
                                                                            momentum=params.p_momentum,
                                                                            factor=params.p_lr_factor,
                                                                            patience=params.p_lr_patience)
        self.entropy_loss = EntropyLoss()
        self.init_target = None
        self.init_loss_fn = nn.CrossEntropyLoss(reduction="elementwise_mean")

    def perm_init_target(self, n_init):
        with torch.no_grad():
            mx = correlation_distribution(self.skip_gram[0].u.weight.data.detach()[:n_init])
            mz = correlation_distribution(self.skip_gram[1].u.weight.data.detach()[:n_init])
            self.init_target = csls_knn(mx, mz)

    def perm_init_step(self):
        self.perm_optimizer.zero_grad()
        batch = torch.LongTensor([self.sampler[0].sample() for _ in range(self.i_bs)])
        for id in [0, 1]:
            for param in self.skip_gram[id].parameters():
                param.requires_grad = False
        x = self.skip_gram[0].u(batch.view(self.i_bs, 1).to(GPU)).view(self.i_bs, -1)
        p = self.perm(x)  # p: Float[bs, n_init]
        loss = self.init_loss_fn(p, self.init_target[batch])
        loss.backward()
        self.perm_optimizer.step()
        return loss.item()

    def perm_step(self, *, fix_embedding=False):
        self.perm_optimizer.zero_grad()
        batch = torch.LongTensor([self.sampler[0].sample() for _ in range(self.p_bs)]).view(self.p_bs, 1).to(GPU)
        for id in [0, 1]:
            for param in self.skip_gram[id].parameters():
                param.requires_grad = not fix_embedding
        x = self.skip_gram[0].u(batch).view(self.p_bs, -1)
        p = self.perm(x)
        e_loss = self.entropy_loss(p)
        p = F.softmax(p, dim=1)
        z = torch.mm(p, self.skip_gram[1].u.weight[:self.p_sample_top])
        gx = torch.einsum("ik,jk->ij", (x, x))
        gz = torch.einsum("ik,jk->ij", (z, z))
        loss = torch.mean((gx - gz) ** 2)
        loss.backward()
        self.perm_optimizer.step()
        return loss.item(), e_loss.item()

    def output(self):
        lst = []
        with torch.no_grad():
            for i in range(0, self.vocab_size_0, self.p_bs):
                batch = torch.arange(i, min(i + self.p_bs, self.vocab_size_0)).view(-1, 1).to(GPU)
                x = self.skip_gram[0].u(batch).view(-1, self.emb_dim)
                y = self.perm(x).topk(10, dim=1, largest=True, sorted=True)[1]  # Long[p_bs, 10]
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

    # Permutation init settings
    parser.add_argument("--i_n_init", type=int, help="number of initializing words")
    parser.add_argument("--i_bs", type=int, help="batch size of initializing")

    # Permutation learning settings
    parser.add_argument("--p_bs", type=int, help="batch size of permutation learning")
    parser.add_argument("--p_lr", type=float, help="learning rate of permutation learning")
    parser.add_argument("--p_lr_factor", type=float, help="lr decay factor of permutation learning")
    parser.add_argument("--p_lr_patience", type=int, help="epochs to wait before decaying lr")
    parser.add_argument("--p_momentum", type=float, help="momentum of permutation learning")
    parser.add_argument("--p_wd", type=float, help="weight decay of permutation learning")
    parser.add_argument("--p_n_units", type=int, help="number of hidden units in permutation model")
    parser.add_argument("--p_bn", action="store_true", help="turn on batch normalization or not")
    parser.add_argument("--p_sample_top", type=int, help="sample top n frequent words in permutation learning")
    parser.add_argument("--p_sample_factor", type=float, help="sample factor of permutation learning")

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
    out_freq = 1000
    trainer.perm_init_target(params.i_n_init)
    step, train_loss, train_norm = 0, 0.0, 0
    for _ in trange(params.n_steps):
        p_loss = trainer.perm_init_step()
        train_loss += p_loss
        train_norm += 1
        if train_norm >= out_freq:
            vis.line(Y=torch.FloatTensor([train_loss / train_norm]), X=torch.LongTensor([step]),
                     win="ip_loss", env=params.out_path, opts={"title": "ip_loss"}, update="append")
            train_norm, train_loss = 0, 0.0
            step += 1
    valid_loss, valid_norm, train_loss, train_norm, train_loss_e = 0.0, 0, 0.0, 0, 0.0
    for epoch in trange(params.n_epochs):
        for _ in trange(params.n_steps):
            p_loss, e_loss = trainer.perm_step(fix_embedding=epoch < params.epoch_tune_emb)
            train_loss += p_loss
            train_loss_e += e_loss
            train_norm += 1
            valid_loss = valid_loss * 0.999 + 0.001 * p_loss
            valid_norm = valid_norm * 0.999 + 0.001 * 1.0
            if train_norm >= out_freq:
                vis.line(Y=torch.FloatTensor([train_loss / train_norm]), X=torch.LongTensor([step]),
                         win="p_loss", env=params.out_path, opts={"title": "p_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([train_loss_e / train_norm]), X=torch.LongTensor([step]),
                         win="e_loss", env=params.out_path, opts={"title": "e_loss"}, update="append")
                train_norm, train_loss, train_loss_e = 0, 0.0, 0.0
                step += 1
        print(f"epoch {epoch} loss is {valid_loss / valid_norm}")
        trainer.scheduler_step(valid_loss)
        valid_loss, valid_norm = 0.0, 0
        dic0, dic1 = convert_dic(corpus_data_0.dic, params.src_lang), convert_dic(corpus_data_1.dic, params.tgt_lang)
        model_output = trainer.output()
        torch.save({"dic0": dic0, "dic1": dic1, "out": model_output}, os.path.join(out_path, f"out-epoch{epoch}.pth"))
    print(params)


if __name__ == '__main__':
    main()
