import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import visdom
from torch.utils.data import DataLoader
from tqdm import trange

import optimizers
from corpus_data_fast_text import CorpusData, concat_collate
from discriminator import Discriminator
from fast_text import FastText
from src.dico_builder import get_candidates, build_dictionary
from src.dictionary import Dictionary
from word_sampler import WordSampler

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")


def _data_queue(corpus_data, *, n_threads, n_sentences, batch_size):
    data_loader = DataLoader(corpus_data, collate_fn=concat_collate, batch_size=n_sentences, num_workers=n_threads,
                             pin_memory=True, shuffle=False)
    while True:
        for u, v in data_loader:
            for i in range(u.shape[0] // batch_size):
                u_b = u[i * batch_size: (i + 1) * batch_size]
                v_b = v[i * batch_size: (i + 1) * batch_size].to(GPU)
                yield u_b, v_b


def _orthogonalize(mapping, beta):
    if beta > 0:
        w = mapping.weight.data
        w.copy_((1 + beta) * w - beta * w.mm(w.transpose(0, 1).mm(w)))


def _csls(x, z, k=10):
    x /= x.norm(p=2, dim=1, keepdim=True).expand_as(x)
    z /= z.norm(p=2, dim=1, keepdim=True).expand_as(z)
    sim = torch.einsum("ik,jk->ij", (x, z))
    sx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)[0].mean(dim=1, keepdim=True).expand_as(sim)
    sz = torch.topk(sim, k=k, dim=0, largest=True, sorted=False)[0].mean(dim=0, keepdim=True).expand_as(sim)
    return sim * 2 - sx - sz


def _procrustes(x, z, p, *, p2=None, mode="S2T"):
    if mode == "S2T":
        m = torch.einsum("ki,kj->ij", (x, z[p]))
    elif mode == "T2S":
        m = torch.einsum("ki,kj->ij", (x[p], z))
    elif mode == "both":
        m = (torch.einsum("ki,kj->ij", (x, z[p])) + torch.einsum("ki,kj->ij", (x[p2], z))) * 0.5
    else:
        raise Exception(f"procrustes mode {mode} does not exist")
    u, _, vt = torch.svd(m)
    return u @ vt


def _refine(x, z, mode="S2T"):
    with torch.no_grad():
        sim = _csls(x, z)
        if mode == "S2T":
            p = sim.argmax(dim=1)
            w = _procrustes(x, z, p, mode="S2T")
        elif mode == "T2S":
            p = sim.argmax(dim=0)
            w = _procrustes(z, x, p, mode="T2S")
        elif mode == "both":
            p1 = sim.argmax(dim=1)
            p2 = sim.argmax(dim=0)
            w = _procrustes(x, z, p1, p2=p2, mode="both")
        else:
            raise Exception(f"procrustes mode {mode} does not exist")
        return x @ w, z


class Trainer:
    def __init__(self, corpus_data_0, corpus_data_1, *, params, n_samples=10000000):
        self.fast_text = [FastText(corpus_data_0.model).to(GPU), FastText(corpus_data_1.model).to(GPU)]
        self.discriminator = Discriminator(params.emb_dim, n_layers=params.d_n_layers, n_units=params.d_n_units,
                                           drop_prob=params.d_drop_prob, drop_prob_input=params.d_drop_prob_input,
                                           leaky=params.d_leaky, batch_norm=params.d_bn).to(GPU)
        self.mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
        self.mapping = self.mapping.to(GPU)
        self.ft_optimizer, self.ft_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_sgd_adapt(self.fast_text[id].parameters(),
                                                            lr=params.ft_lr, mode="max")
            self.ft_optimizer.append(optimizer)
            self.ft_scheduler.append(scheduler)
        self.a_optimizer, self.a_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_sgd_adapt(
                [{"params": self.fast_text[id].u.parameters()}, {"params": self.fast_text[id].v.parameters()}],
                lr=params.a_lr, mode="max")
            self.a_optimizer.append(optimizer)
            self.a_scheduler.append(scheduler)
        if params.d_optimizer == "SGD":
            self.d_optimizer, self.d_scheduler = optimizers.get_sgd_adapt(self.discriminator.parameters(),
                                                                          lr=params.d_lr, mode="max", wd=params.d_wd)

        elif params.d_optimizer == "RMSProp":
            self.d_optimizer, self.d_scheduler = optimizers.get_rmsprop_linear(self.discriminator.parameters(),
                                                                               params.n_steps,
                                                                               lr=params.d_lr, wd=params.d_wd)
        else:
            raise Exception(f"Optimizer {params.d_optimizer} not found.")
        if params.m_optimizer == "SGD":
            self.m_optimizer, self.m_scheduler = optimizers.get_sgd_adapt(self.mapping.parameters(),
                                                                          lr=params.m_lr, mode="max", wd=params.m_wd)
        elif params.m_optimizer == "RMSProp":
            self.m_optimizer, self.m_scheduler = optimizers.get_rmsprop_linear(self.mapping.parameters(),
                                                                               params.n_steps,
                                                                               lr=params.m_lr, wd=params.m_wd)
        else:
            raise Exception(f"Optimizer {params.m_optimizer} not found")
        self.m_beta = params.m_beta
        self.smooth = params.smooth
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        self.corpus_data_queue = [
            _data_queue(corpus_data_0, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.ft_bs),
            _data_queue(corpus_data_1, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.ft_bs)
        ]
        self.sampler = [
            WordSampler(corpus_data_0.dic, n_urns=n_samples, alpha=params.a_sample_factor, top=params.a_sample_top),
            WordSampler(corpus_data_1.dic, n_urns=n_samples, alpha=params.a_sample_factor, top=params.a_sample_top)]
        self.d_bs = params.d_bs
        self.dic_0, self.dic_1 = corpus_data_0.dic, corpus_data_1.dic
        self.r_top = params.r_top
        self.r_bs = params.r_top if params.r_bs == 0 else params.r_bs

    def fast_text_step(self):
        losses = []
        for id in [0, 1]:
            self.ft_optimizer[id].zero_grad()
            u_b, v_b = self.corpus_data_queue[id].__next__()
            s = self.fast_text[id](u_b, v_b)
            loss = FastText.loss_fn(s)
            loss.backward()
            self.ft_optimizer[id].step()
            losses.append(loss.item())
        return losses[0], losses[1]

    def get_adv_batch(self, *, reverse, fix_embedding=False):
        batch = [[self.sampler[id].sample() for _ in range(self.d_bs)]
                 for id in [0, 1]]
        batch = [self.fast_text[id].model.get_bag(batch[id], self.fast_text[id].u.weight.device)
                 for id in [0, 1]]
        if fix_embedding:
            with torch.no_grad():
                x = [self.fast_text[id].u(batch[id][0], batch[id][1]).view(self.d_bs, -1) for id in [0, 1]]
        else:
            x = [self.fast_text[id].u(batch[id][0], batch[id][1]).view(self.d_bs, -1) for id in [0, 1]]
        x[0] = self.mapping(x[0])
        x = torch.cat(x, 0)
        y = torch.FloatTensor(self.d_bs * 2).to(GPU).uniform_(0.0, self.smooth)
        if reverse:
            y[: self.d_bs] = 1 - y[: self.d_bs]
        else:
            y[self.d_bs:] = 1 - y[self.d_bs:]
        return x, y

    def adversarial_step(self, fix_embedding=False):
        for id in [0, 1]:
            self.a_optimizer[id].zero_grad()
        self.m_optimizer.zero_grad()
        self.discriminator.eval()
        x, y = self.get_adv_batch(reverse=True, fix_embedding=fix_embedding)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        for id in [0, 1]:
            self.a_optimizer[id].step()
        self.m_optimizer.step()
        _orthogonalize(self.mapping, self.m_beta)
        return loss.item()

    def discriminator_step(self):
        self.d_optimizer.zero_grad()
        self.discriminator.train()
        with torch.no_grad():
            x, y = self.get_adv_batch(reverse=False)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

    def scheduler_step(self, metric):
        for id in [0, 1]:
            self.ft_scheduler[id].step(metric)
            self.a_scheduler[id].step(metric)
        # self.d_scheduler.step(metric)
        self.m_scheduler.step(metric)


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
    parser.add_argument("data_path_0", type=str, help="path of data 0")
    parser.add_argument("data_path_1", type=str, help="path of data 1")
    parser.add_argument("model_path_0", type=str, help="path of fast-text model 0")
    parser.add_argument("model_path_1", type=str, help="path of fast-text model 1")
    parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
    parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--checkpoint", action="store_true", help="save a checkpoint after each epoch")
    parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
    parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")

    # Global settings
    parser.add_argument("--src_lang", type=str, help="language of embedding 0")
    parser.add_argument("--tgt_lang", type=str, help="language of embedding 1")
    parser.add_argument("--vocab_size", type=int, help="size of output vocabulary")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--n_steps", type=int, help="number of steps per epoch")
    parser.add_argument("--epoch_adv", type=int, help="the epoch to start adversarial training for embeddings")
    parser.add_argument("--epoch_ft", type=int, help="the epoch to start skip-gram training")
    parser.add_argument("--interval_ft", type=int, help="the interval of training skip-gram")
    parser.add_argument("--smooth", type=float, help="label smooth for adversarial training")
    parser.add_argument("--normalize_post", type=str, default="", help="how to normalize the embedding after training")

    # Skip-gram settings
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--n_threads", type=int, help="number of data loader threads")
    parser.add_argument("--ft_bs", type=int, help="batch size")
    parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
    parser.add_argument("--ft_lr", type=float, help="initial learning rate of skip-gram")
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

    # Refinement settings
    parser.add_argument("--r_top", type=int, help="only sample top n words to refine")
    parser.add_argument("--r_bs", type=int, default=0, help="batch size for refinement (0 for r_top)")
    parser.add_argument("--r_mode", type=str, default="S2T", help="mode for refinement (S2T, T2S, both)")
    parser.add_argument("--r_n_steps", type=int, help="number of refinement steps")

    params = parser.parse_args()

    print(params)

    out_path = os.path.join(params.modelDir, params.out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    corpus_data_0 = CorpusData(os.path.join(params.dataDir, params.data_path_0),
                               os.path.join(params.dataDir, params.model_path_0),
                               max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
    corpus_data_1 = CorpusData(os.path.join(params.dataDir, params.data_path_1),
                               os.path.join(params.dataDir, params.model_path_1),
                               max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
    trainer = Trainer(corpus_data_0, corpus_data_1, params=params)
    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(out_path, "log.txt"), use_incoming_socket=False)
    out_freq = 500
    step, c, ft_loss, d_loss, a_loss, best_valid_metric = 0, 0, [0.0, 0.0], 0.0, 0.0, 0.0
    best_x, best_z = None, None
    dic0, dic1 = convert_dic(corpus_data_0.dic, params.src_lang), convert_dic(corpus_data_1.dic, params.tgt_lang)
    for epoch in trange(params.n_epochs):
        for _ in trange(params.n_steps):
            if epoch >= params.epoch_ft and c % params.interval_ft == 0:
                l0, l1 = trainer.fast_text_step()
                ft_loss[0] += l0
                ft_loss[1] += l1
            d_loss += sum([trainer.discriminator_step() for _ in range(params.d_n_steps)]) / params.d_n_steps
            a_loss += trainer.adversarial_step(fix_embedding=epoch < params.epoch_adv)
            c += 1
            if c >= out_freq:
                vis.line(Y=torch.FloatTensor([ft_loss[0] / c * params.interval_ft]), X=torch.LongTensor([step]),
                         win="ft_loss_0", env=params.out_path, opts={"title": "ft_loss_0"}, update="append")
                vis.line(Y=torch.FloatTensor([ft_loss[1] / c * params.interval_ft]), X=torch.LongTensor([step]),
                         win="ft_loss_1", env=params.out_path, opts={"title": "ft_loss_1"}, update="append")
                vis.line(Y=torch.FloatTensor([d_loss / c]), X=torch.LongTensor([step]),
                         win="d_loss", env=params.out_path, opts={"title": "d_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([a_loss / c]), X=torch.LongTensor([step]),
                         win="a_loss", env=params.out_path, opts={"title": "a_loss"}, update="append")
                c, ft_loss, d_loss, a_loss = 0, [0.0, 0.0], 0.0, 0.0
                step += 1
        with torch.no_grad():
            emb0 = trainer.fast_text[0].get_input_matrix(trainer.dic_0, params.vocab_size, params.ft_bs)
            emb1 = trainer.fast_text[1].get_input_matrix(trainer.dic_1, params.vocab_size, params.ft_bs)
            emb0 = trainer.mapping(emb0)
        valid_metric = dist_mean_cosine(emb0, emb1)
        trainer.scheduler_step(valid_metric)
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            best_x, best_z = emb0, emb1
            emb0 = normalize_embeddings(emb0, params.normalize_post)
            emb1 = normalize_embeddings(emb1, params.normalize_post)
            torch.save({"dico": dic0, "vectors": emb0}, os.path.join(out_path, f"{params.src_lang}-epoch{epoch}.pth"))
            torch.save({"dico": dic1, "vectors": emb1}, os.path.join(out_path, f"{params.tgt_lang}-epoch{epoch}.pth"))
    x, z = best_x, best_z
    for i in trange(params.r_n_steps):
        x, z = _refine(x, z, mode=params.r_mode)
        valid_metric = dist_mean_cosine(x, z)
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            emb0 = normalize_embeddings(x, params.normalize_post)
            emb1 = normalize_embeddings(z, params.normalize_post)
            torch.save({"dico": dic0, "vectors": emb0}, os.path.join(out_path, f"{params.src_lang}-refinement{i}.pth"))
            torch.save({"dico": dic1, "vectors": emb1}, os.path.join(out_path, f"{params.tgt_lang}-refinement{i}.pth"))
    print(params)


if __name__ == "__main__":
    main()
