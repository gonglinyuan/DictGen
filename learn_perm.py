import datetime
import argparse
import os
from datetime import datetime

import fastText
import torch
import torch.nn as nn
import visdom
from torch.utils.data import DataLoader
from tqdm import trange

import optimizers
from corpus_data import concat_collate, BlockRandomSampler, CorpusData
from skip_gram import SkipGram
from src.dico_builder import get_candidates, build_dictionary
from src.dictionary import Dictionary
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
        self.perm_optimizer, self.perm_scheduler = optimizers.get_sgd_adapt(self.perm.parameters(),
                                                                            lr=params.p_lr, wd=params.p_wd)

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
                x[0] = torch.mm(x[0], self.skip_gram[1].u.weight)
        else:
            x[0] = torch.mm(x[0], self.skip_gram[1].u.weight)
        x = [torch.einsum("ik,jk->ij", (x[id], x[id])) for id in [0, 1]]
        loss = torch.mean((x[0] - x[1]) ** 2)
        loss.backward()
        self.perm_optimizer.step()
        return loss.item()

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

def dist_mean_cosine(src_emb, tgt_emb):
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).clamp(min=1e-3).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).clamp(min=1e-3).expand_as(tgt_emb)
    dico_method = 'csls_knn_10'
    dico_build = 'S2T'
    dico_max_size = 10000

    class Dummy:
        def __init__(self):
            self.dico_eval = "default"
            self.dico_method = "csls_knn_10"
            self.dico_build = "S2T"
            self.dico_threshold = 0
            self.dico_max_rank = 15000
            self.dico_min_size = 0
            self.dico_max_size = 0
            self.cuda = True

    _params = Dummy()
    _params.dico_method = dico_method
    _params.dico_build = dico_build
    _params.dico_threshold = 0
    _params.dico_max_rank = 10000
    _params.dico_min_size = 0
    _params.dico_max_size = dico_max_size
    s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
    t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
    dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
    if dico is None:
        mean_cosine = -1e9
    else:
        mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
    mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch.Tensor) else mean_cosine
    print("Mean cosine (%s method, %s build, %i max size): %.5f"
          % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
    return mean_cosine

def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description="adversarial training")

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
    parser.add_argument("--checkpoint", action="store_true", help="save a checkpoint after each epoch")
    parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
    parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")

    # Global settings
    parser.add_argument("--src_lang", type=str, help="language of embedding 0")
    parser.add_argument("--tgt_lang", type=str, help="language of embedding 1")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--n_steps", type=int, help="number of steps per epoch")
    parser.add_argument("--epoch_adv", type=int, help="the epoch to start adversarial training for embeddings")
    parser.add_argument("--epoch_sg", type=int, help="the epoch to start skip-gram training")
    parser.add_argument("--interval_sg", type=int, help="the interval of training skip-gram")
    parser.add_argument("--smooth", type=float, help="label smooth for adversarial training")
    parser.add_argument("--normalize_pre", type=str, default="", help="how to normalize the embedding before training")
    parser.add_argument("--normalize_post", type=str, default="", help="how to normalize the embedding after training")

    # Skip-gram settings
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--n_threads", type=int, help="number of data loader threads")
    parser.add_argument("--sg_bs", type=int, help="batch size")
    parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
    parser.add_argument("--sg_lr", type=float, help="initial learning rate of skip-gram")
    parser.add_argument("--threshold", type=float, default=1e-4, help="sampling threshold")

    # Discriminator settings
    parser.add_argument("--d_n_layers", type=int, help="number of hidden layers of the discriminator")
    parser.add_argument("--d_n_units", type=int, help="number of units per hidden layer of the discriminator")
    parser.add_argument("--d_drop_prob", type=float, help="dropout probability after each layer of the discriminator")
    parser.add_argument("--d_drop_prob_input", type=float, help="dropout probability of the input of the discriminator")
    parser.add_argument("--d_leaky", type=float, help="slope of leaky ReLU of the discriminator")
    parser.add_argument("--d_bs", type=int, help="batch size for the discriminator")
    parser.add_argument("--d_lr", type=float, help="max learning rate of adversarial training for the discriminator")
    parser.add_argument("--d_wd", type=float, help="weight decay of adversarial training for the discriminator")
    parser.add_argument("--d_bn", action="store_true", help="turn on batch normalization for the discriminator or not")
    parser.add_argument("--d_optimizer", type=str, help="optimizer for the discriminator")
    parser.add_argument("--d_n_steps", type=int, help="number of discriminator steps per interation")

    # Mapping settings
    parser.add_argument("--m_optimizer", type=str, help="optimizer for the mapping")
    parser.add_argument("--m_lr", type=float, help="max learning rate for the mapping")
    parser.add_argument("--m_wd", type=float, help="weight decay for the mapping")
    parser.add_argument("--m_beta", type=float, help="beta to orthogonalize the mapping")

    # Adversarial training settings
    parser.add_argument("--a_lr", type=float, help="max learning rate of adversarial training for embeddings")
    parser.add_argument("--a_sample_top", type=int, default=0, help="only sample top n words in adversarial training")
    parser.add_argument("--a_sample_factor", type=float, help="sample factor in adversarial training")

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
    step, c, sg_loss, d_loss, a_loss = 0, 0, [0.0, 0.0], 0.0, 0.0
    for epoch in trange(params.n_epochs):
        for _ in trange(params.n_steps):
            if epoch >= params.epoch_sg and c % params.interval_sg == 0:
                l0, l1 = trainer.skip_gram_step()
                sg_loss[0] += l0
                sg_loss[1] += l1
            d_loss += sum([trainer.discriminator_step() for _ in range(params.d_n_steps)]) / params.d_n_steps
            a_loss += trainer.adversarial_step(fix_embedding=epoch < params.epoch_adv)
            c += 1
            if c >= out_freq:
                vis.line(Y=torch.FloatTensor([sg_loss[0] / c * params.interval_sg]), X=torch.LongTensor([step]),
                         win="sg_loss_0", env=params.out_path, opts={"title": "sg_loss_0"}, update="append")
                vis.line(Y=torch.FloatTensor([sg_loss[1] / c * params.interval_sg]), X=torch.LongTensor([step]),
                         win="sg_loss_1", env=params.out_path, opts={"title": "sg_loss_1"}, update="append")
                vis.line(Y=torch.FloatTensor([d_loss / c]), X=torch.LongTensor([step]),
                         win="d_loss", env=params.out_path, opts={"title": "d_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([a_loss / c]), X=torch.LongTensor([step]),
                         win="a_loss", env=params.out_path, opts={"title": "a_loss"}, update="append")
                c, sg_loss, d_loss, a_loss = 0, [0.0, 0.0], 0.0, 0.0
                step += 1
        emb0, emb1 = trainer.skip_gram[0].u.weight.data.detach()[:-1], trainer.skip_gram[1].u.weight.data.detach()[:-1]
        with torch.no_grad():
            emb0 = trainer.mapping(emb0)
        trainer.scheduler_step(dist_mean_cosine(emb0, emb1))
        dic0, dic1 = convert_dic(corpus_data_0.dic, params.src_lang), convert_dic(corpus_data_1.dic, params.tgt_lang)
        emb0 = normalize_embeddings(emb0, params.normalize_post)
        emb1 = normalize_embeddings(emb1, params.normalize_post)
        torch.save({"dico": dic0, "vectors": emb0}, os.path.join(out_path, f"{params.src_lang}-epoch{epoch}.pth"))
        torch.save({"dico": dic1, "vectors": emb1}, os.path.join(out_path, f"{params.tgt_lang}-epoch{epoch}.pth"))
    print(params)