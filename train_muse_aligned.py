import argparse
import os
from datetime import datetime

import fastText
import numpy as np
import scipy.linalg
import torch
import torch.autograd as autograd
import torch.nn as nn
import visdom
from tqdm import trange

import optimizers
from corpus_data_fast_text import CorpusData
from discriminator import Discriminator
from src.dico_builder import get_candidates, build_dictionary
from src.dictionary import Dictionary
from word_sampler import WordSampler

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")


def _csls_nn(x, z, *, bs, k=10):
    x = x / x.norm(p=2, dim=1, keepdim=True).expand_as(x)
    z = z / z.norm(p=2, dim=1, keepdim=True).expand_as(z)
    n = x.shape[0]
    sz = torch.FloatTensor(1, n).to(GPU)
    p = torch.LongTensor(n).to(GPU)
    for i in range(0, n, bs):
        sim = torch.einsum("ik,jk->ij", (x, z[i:min(i + bs, n), :]))
        sz[:, i:min(i + bs, n)] = torch.topk(sim, k=k, dim=0, largest=True, sorted=False)[0].mean(dim=0, keepdim=True)
    for i in range(0, n, bs):
        sim = torch.einsum("ik,jk->ij", (x[i:min(i + bs, n), :], z))  # Float[bs, n]
        sx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)[0].mean(dim=1, keepdim=True)  # Float[bs, 1]
        csls = sim * 2 - sx.expand_as(sim) - sz.expand_as(sim)
        p[i:min(i + bs, n)] = torch.argmax(csls, dim=1)
    return p


def _orthogonal_project(m):
    u, _, vt = scipy.linalg.svd(m)
    return u @ vt


def _wasserstein_distance(y_hat, y):
    return torch.mean(y_hat * (y * 2 - 1))


def _spectral_alignment(x, z):
    x, z = np.array(x, dtype=np.float64), np.array(z, dtype=np.float64)
    u1, s1, v1t = scipy.linalg.svd(x, full_matrices=False)
    u2, s2, v2t = scipy.linalg.svd(z, full_matrices=False)
    s = (s1 + s2) * 0.5
    x = np.einsum("ik,k,kj->ij", u1, s, v1t)
    z = np.einsum("ik,k,kj->ij", u2, s, v2t)
    x = torch.from_numpy(x).to(torch.float).to(GPU)
    z = torch.from_numpy(z).to(torch.float).to(GPU)
    return x, z


def _refine(x, z, top, bs, mode="S2T"):
    with torch.no_grad():
        if mode == "S2T":
            p = _csls_nn(x, z, bs=bs)  # Long[n]
            p = torch.stack((torch.arange(x.shape[0], device=GPU), p), dim=1)  # Long[n, 2]
            p = p.masked_select((p.max(dim=1, keepdim=True)[0] < top).expand_as(p)).view(-1, 2)  # Long[?, 2]
            xx, zz = np.array(x[p[:, 0]], dtype=np.float64), np.array(z[p[:, 1]], dtype=np.float64)
            m = np.einsum("ki,kj->ij", xx, zz)
            w = _orthogonal_project(m)
        elif mode == "T2S":
            p = _csls_nn(z, x, bs=bs)  # Long[n]
            p = torch.stack((torch.arange(x.shape[0], device=GPU), p), dim=1)  # Long[n, 2]
            p = p.masked_select((p.max(dim=1, keepdim=True)[0] < top).expand_as(p)).view(-1, 2)  # Long[?, 2]
            xx, zz = np.array(x[p[:, 1]], dtype=np.float64), np.array(z[p[:, 0]], dtype=np.float64)
            m = np.einsum("ki,kj->ij", xx, zz)
            w = _orthogonal_project(m)
        elif mode == "both":
            p = _csls_nn(x, z, bs=bs)  # Long[n]
            p = torch.stack((torch.arange(x.shape[0], device=GPU), p), dim=1)  # Long[n, 2]
            p = p.masked_select((p.max(dim=1, keepdim=True)[0] < top).expand_as(p)).view(-1, 2)  # Long[?, 2]
            xx, zz = np.array(x[p[:, 0]], dtype=np.float64), np.array(z[p[:, 1]], dtype=np.float64)
            m1 = np.einsum("ki,kj->ij", xx, zz)
            p = _csls_nn(z, x, bs=bs)  # Long[n]
            p = torch.stack((torch.arange(x.shape[0], device=GPU), p), dim=1)  # Long[n, 2]
            p = p.masked_select((p.max(dim=1, keepdim=True)[0] < top).expand_as(p)).view(-1, 2)  # Long[?, 2]
            xx, zz = np.array(x[p[:, 1]], dtype=np.float64), np.array(z[p[:, 0]], dtype=np.float64)
            m2 = np.einsum("ki,kj->ij", xx, zz)
            w = _orthogonal_project((m1 + m2) * 0.5)
        else:
            raise Exception(f"procrustes mode {mode} does not exist")
        return x @ torch.from_numpy(w).to(torch.float).to(GPU), z


class Mapping(nn.Module):
    def __init__(self, emb_dim):
        super(Mapping, self).__init__()
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.Tensor(self.emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.copy_(torch.ones(self.emb_dim))

    def forward(self, x):
        return x * self.weight.expand_as(x)

    def clip_weights(self):
        self.weight.data.clamp_(-1.0, 1.0)


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


def normalize_embeddings_np(emb, types, mean=None):
    eps = 1e-4
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = np.mean(emb, axis=0, keepdims=True)
            return emb - mean
        elif t == 'renorm':
            return emb / (np.linalg.norm(emb, ord=2, axis=1, keepdims=True) + eps)
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return emb


class Trainer:
    def __init__(self, params, *, n_samples=10000000):
        self.model = [fastText.load_model(params.model_path_0), fastText.load_model(params.model_path_1)]
        self.dic = [list(zip(*self.model[id].get_words(include_freq=True))) for id in [0, 1]]
        x = [np.empty((params.vocab_size, params.emb_dim), dtype=np.float64) for _ in [0, 1]]
        for id in [0, 1]:
            for i in range(params.vocab_size):
                x[id][i, :] = self.model[id].get_word_vector(self.dic[id][0])
            x[id] = normalize_embeddings_np(x[id], params.normalize_pre)
        u0, s0, _ = scipy.linalg.svd(x[0], full_matrices=False)
        u1, s1, _ = scipy.linalg.svd(x[1], full_matrices=False)
        if params.spectral_align_pre:
            s = (s0 + s1) * 0.5
            x[0] = u0 @ np.diag(s)
            x[1] = u1 @ np.diag(s)
        else:
            x[0] = u0 @ np.diag(s0)
            x[1] = u1 @ np.diag(s1)
        self.embedding = [nn.Embedding.from_pretrained(x[id], freeze=True, sparse=True) for id in [0, 1]]
        self.discriminator = Discriminator(params.emb_dim, n_layers=params.d_n_layers, n_units=params.d_n_units,
                                           drop_prob=params.d_drop_prob, drop_prob_input=params.d_drop_prob_input,
                                           leaky=params.d_leaky, batch_norm=params.d_bn).to(GPU)
        self.mapping = Mapping(params.emb_dim).to(GPU)
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
                                                                          lr=params.m_lr, mode="max", wd=params.m_wd,
                                                                          factor=params.m_lr_decay,
                                                                          patience=params.m_lr_patience)
        elif params.m_optimizer == "RMSProp":
            self.m_optimizer, self.m_scheduler = optimizers.get_rmsprop_linear(self.mapping.parameters(),
                                                                               params.n_steps,
                                                                               lr=params.m_lr, wd=params.m_wd)
        else:
            raise Exception(f"Optimizer {params.m_optimizer} not found")
        self.m_beta = params.m_beta
        self.smooth = params.smooth
        self.wgan = params.wgan
        self.d_clip_mode = params.d_clip_mode
        if params.wgan:
            self.loss_fn = _wasserstein_distance
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        self.sampler = [WordSampler(self.dic[id], n_urns=n_samples, alpha=params.a_sample_factor,
                                    top=params.a_sample_top)
                        for id in [0, 1]]
        self.d_bs = params.d_bs
        self.d_gp = params.d_gp

    def get_adv_batch(self, *, reverse, gp=False):
        batch = [torch.LongTensor([self.sampler[id].sample() for _ in range(self.d_bs)]).view(self.d_bs, 1).to(GPU)
                 for id in [0, 1]]
        with torch.no_grad():
            x = [self.embedding[id](batch[id]).view(self.d_bs, -1) for id in [0, 1]]
        y = torch.FloatTensor(self.d_bs * 2).to(GPU).uniform_(0.0, self.smooth)
        if reverse:
            y[: self.d_bs] = 1 - y[: self.d_bs]
        else:
            y[self.d_bs:] = 1 - y[self.d_bs:]
        x[0] = self.mapping(x[0])
        if gp:
            t = torch.FloatTensor(self.d_bs, 1).to(GPU).uniform_(0.0, 1.0).expand_as(x[0])
            z = x[0] * t + x[1] * (1.0 - t)
            x = torch.cat(x, 0)
            return x, y, z
        else:
            x = torch.cat(x, 0)
            return x, y

    def adversarial_step(self):
        self.m_optimizer.zero_grad()
        self.discriminator.eval()
        x, y = self.get_adv_batch(reverse=True)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.m_optimizer.step()
        self.mapping.clip_weights()
        return loss.item()

    def discriminator_step(self):
        self.d_optimizer.zero_grad()
        self.discriminator.train()
        with torch.no_grad():
            if self.d_gp > 0:
                x, y, z = self.get_adv_batch(reverse=False, gp=True)
            else:
                x, y = self.get_adv_batch(reverse=False)
                z = None
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        if self.d_gp > 0:
            z.requires_grad_()
            z_out = self.discriminator(z)
            g = autograd.grad(z_out, z, grad_outputs=torch.ones_like(z_out, device=GPU),
                              retain_graph=True, create_graph=True, only_inputs=True)[0]
            gp = torch.mean((g.norm(p=2, dim=1) - 1.0) ** 2)
            loss += self.d_gp * gp
        loss.backward()
        self.d_optimizer.step()
        if self.wgan:
            self.discriminator.clip_weights(self.d_clip_mode)
        return loss.item()

    def scheduler_step(self, metric):
        self.m_scheduler.step(metric)


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
    parser.add_argument("--smooth", type=float, help="label smooth for adversarial training")
    parser.add_argument("--normalize_pre", type=str, default="", help="how to normalize the embedding before training")
    parser.add_argument("--normalize_mid", type=str, default="",
                        help="how to normalize the embedding before refinement")
    parser.add_argument("--normalize_post", type=str, default="", help="how to normalize the embedding after training")
    parser.add_argument("--spectral_align_pre", action="store_true", help="spectral align before adv training")
    parser.add_argument("--spectral_align_mid", action="store_true", help="spectral align before refinement")
    # parser.add_argument("--spectral_align_post", action="store_true", help="spectral align after refinement")

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
    parser.add_argument("--d_clip_mode", type=str, default="none", help="how to clip weights")
    parser.add_argument("--d_gp", type=float, default=-1, help="the gradient penalty factor (-1 to disable)")

    # WGAN settings
    parser.add_argument("--wgan", action="store_true", help="use WGAN or ordinary GAN")

    # Mapping settings
    parser.add_argument("--m_optimizer", type=str, help="optimizer for the mapping")
    parser.add_argument("--m_lr", type=float, help="max learning rate for the mapping")
    parser.add_argument("--m_lr_decay", type=float, help="learning rate decay factor")
    parser.add_argument("--m_lr_patience", type=float, help="learning rate decay patience")
    parser.add_argument("--m_wd", type=float, help="weight decay for the mapping")
    parser.add_argument("--m_beta", type=float, help="beta to orthogonalize the mapping")

    # Adversarial training settings
    parser.add_argument("--a_sample_top", type=int, default=0, help="only sample top n words in adversarial training")
    parser.add_argument("--a_sample_factor", type=float, help="sample factor in adversarial training")

    # Refinement settings
    parser.add_argument("--r_top", type=int, help="only sample top n words to refine")
    parser.add_argument("--r_bs", type=int, help="batch size for refinement")
    parser.add_argument("--r_mode", type=str, default="S2T", help="mode for refinement (S2T, T2S, both)")
    parser.add_argument("--r_n_steps", type=int, help="number of refinement steps")

    params = parser.parse_args()

    print(params)

    out_path = os.path.join(params.modelDir, params.out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    trainer = Trainer(params)
    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(out_path, "log.txt"), use_incoming_socket=False)
    out_freq = 500
    step, c, ft_loss, d_loss, a_loss, best_valid_metric = 0, 0, [0.0, 0.0], 0.0, 0.0, 0.0
    best_x, best_z = None, None
    dic0, dic1 = convert_dic(trainer.dic[0], params.src_lang), convert_dic(trainer.dic[1], params.tgt_lang)
    for epoch in trange(params.n_epochs):
        for _ in trange(params.n_steps):
            d_loss += sum([trainer.discriminator_step() for _ in range(params.d_n_steps)]) / params.d_n_steps
            a_loss += trainer.adversarial_step()
            c += 1
            if c >= out_freq:
                vis.line(Y=torch.FloatTensor([d_loss / c]), X=torch.LongTensor([step]),
                         win="d_loss", env=params.out_path, opts={"title": "d_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([a_loss / c]), X=torch.LongTensor([step]),
                         win="a_loss", env=params.out_path, opts={"title": "a_loss"}, update="append")
                c, ft_loss, d_loss, a_loss = 0, [0.0, 0.0], 0.0, 0.0
                step += 1
        with torch.no_grad():
            emb0 = trainer.embedding[0].weight.data.clone().detach()
            emb1 = trainer.embedding[1].weight.data.clone().detach()
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
    x = normalize_embeddings(best_x, params.normalize_mid)
    z = normalize_embeddings(best_z, params.normalize_mid)
    # torch.save([x, z], os.path.join("tmp.pth"))
    if params.spectral_align_mid:
        x, z = _spectral_alignment(x, z)
    for i in trange(params.r_n_steps):
        x, z = _refine(x, z, top=params.r_top, bs=params.r_bs, mode=params.r_mode)
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
