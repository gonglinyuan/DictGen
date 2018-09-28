import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader
from tqdm import trange

from corpus_data_fast_text import CorpusData, concat_collate
from discriminator import Discriminator
from fast_text import FastText
from src.dictionary import Dictionary

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


class Trainer:
    def __init__(self, corpus_data, *, params):
        self.fast_text = FastText(corpus_data.model).to(GPU)
        self.discriminator = Discriminator(params.emb_dim, n_layers=params.d_n_layers, n_units=params.d_n_units,
                                           drop_prob=params.d_drop_prob, drop_prob_input=params.d_drop_prob_input,
                                           leaky=params.d_leaky, batch_norm=params.d_bn).to(GPU)
        self.ft_optimizer = optim.SGD(self.fast_text.parameters(), lr=params.ft_lr)
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=params.d_lr, weight_decay=params.d_wd)
        self.a_optimizer = optim.SGD([{"params": self.fast_text.u.parameters()},
                                      {"params": self.fast_text.v.parameters()}], lr=params.a_lr)
        self.smooth = params.smooth
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        self.corpus_data_queue = _data_queue(corpus_data, n_threads=params.n_threads, n_sentences=params.n_sentences,
                                             batch_size=params.ft_bs)
        self.vocab_size = params.vocab_size
        self.d_bs = params.d_bs
        self.split = params.split
        self.align_output = params.align_output

    def fast_text_step(self):
        self.ft_optimizer.zero_grad()
        u_b, v_b = self.corpus_data_queue.__next__()
        s = self.fast_text(u_b, v_b)
        loss = FastText.loss_fn(s)
        loss.backward()
        self.ft_optimizer.step()
        return loss.item()

    def get_adv_batch(self, *, reverse, fix_embedding):
        vocab_split, bs_split = int(self.vocab_size * self.split), int(self.d_bs * self.split)
        x = (torch.randint(0, vocab_split, size=(bs_split,), dtype=torch.long).tolist() +
             torch.randint(vocab_split, self.vocab_size, size=(self.d_bs - bs_split,), dtype=torch.long).tolist())
        if self.align_output:
            x = torch.LongTensor(x).view(self.d_bs, 1).to(GPU)
            if fix_embedding:
                with torch.no_grad():
                    x = self.fast_text.v(x).view(self.d_bs, -1)
            else:
                x = self.fast_text.v(x).view(self.d_bs, -1)
        else:
            x = self.fast_text.model.get_bag(x, self.fast_text.u.weight.device)
            if fix_embedding:
                with torch.no_grad():
                    x = self.fast_text.u(x[0], x[1]).view(self.d_bs, -1)
            else:
                x = self.fast_text.u(x[0], x[1]).view(self.d_bs, -1)
        y = torch.FloatTensor(self.d_bs).to(GPU).uniform_(0.0, self.smooth)
        if reverse:
            y[:bs_split] = 1 - y[:bs_split]
        else:
            y[bs_split:] = 1 - y[bs_split:]
        return x, y

    def discriminator_step(self):
        self.d_optimizer.zero_grad()
        self.discriminator.train()
        with torch.no_grad():
            x, y = self.get_adv_batch(reverse=False, fix_embedding=True)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

    def adversarial_step(self):
        self.a_optimizer.zero_grad()
        self.discriminator.eval()
        x, y = self.get_adv_batch(reverse=True, fix_embedding=False)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.a_optimizer.step()
        return loss.item()


def convert_dic(dic, lang):
    id2word, word2id = {}, {}
    for i in range(len(dic)):
        id2word[i] = dic[i][0]
        word2id[dic[i][0]] = i
    return Dictionary(id2word, word2id, lang)


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description="adversarial training")

    # Paths
    parser.add_argument("data_path", type=str, help="path of data")
    parser.add_argument("model_path", type=str, help="path of fast-text model")
    parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
    parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--checkpoint", action="store_true", help="save a checkpoint after each epoch")
    parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
    parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")

    # Global settings
    parser.add_argument("--lang", type=str, help="language")
    parser.add_argument("--vocab_size", type=int, help="size of vocabulary to sample")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--n_steps", type=int, help="number of steps per epoch")
    parser.add_argument("--smooth", type=float, help="label smooth for adversarial training")
    parser.add_argument("--split", type=float, help="split ratio for adversarial training")
    parser.add_argument("--align_output", action="store_true", help="align output embeddings")

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
    parser.add_argument("--d_n_steps", type=int, help="number of discriminator steps per interation")

    # Adversarial training settings
    parser.add_argument("--a_lr", type=float, help="max learning rate of adversarial training for embeddings")

    params = parser.parse_args()

    print(params)

    out_path = os.path.join(params.modelDir, params.out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    corpus_data = CorpusData(os.path.join(params.dataDir, params.data_path),
                             os.path.join(params.dataDir, params.model_path),
                             max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
    trainer = Trainer(corpus_data, params=params)
    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(out_path, "log.txt"), use_incoming_socket=False)
    dico = convert_dic(corpus_data.dic, params.lang)
    out_freq = 500
    c, ft_loss, d_loss, a_loss, step = 0, 0.0, 0.0, 0.0, 0
    for epoch in range(params.n_epochs):
        for _ in trange(params.n_steps):
            ft_loss += trainer.fast_text_step()
            d_loss += sum([trainer.discriminator_step() for _ in range(params.d_n_steps)]) / params.d_n_steps
            a_loss += trainer.adversarial_step()
            c += 1
            if c >= out_freq:
                vis.line(Y=torch.FloatTensor([ft_loss / c]), X=torch.LongTensor([step]),
                         win="ft_loss", env=params.out_path, opts={"title": "ft_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([d_loss / c]), X=torch.LongTensor([step]),
                         win="d_loss", env=params.out_path, opts={"title": "d_loss"}, update="append")
                vis.line(Y=torch.FloatTensor([a_loss / c]), X=torch.LongTensor([step]),
                         win="a_loss", env=params.out_path, opts={"title": "a_loss"}, update="append")
                c, ft_loss, d_loss, a_loss = 0, 0.0, 0.0, 0.0
                step += 1
        with torch.no_grad():
            emb = trainer.fast_text.get_input_matrix(params.vocab_size, params.ft_bs, CPU)
        if params.checkpoint:
            torch.save({
                "dico": dico,
                "vectors": emb,
                "state_dict": trainer.fast_text.state_dict()
            }, os.path.join(out_path, f"epoch{epoch}.pth"))
    print(params)


if __name__ == "__main__":
    main()
