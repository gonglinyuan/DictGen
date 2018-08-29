import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import visdom
from torch.utils.data import DataLoader
from tqdm import trange

import optimizers
from corpus_data import concat_collate, BlockRandomSampler, CorpusData
from discriminator import Discriminator
from skip_gram import SkipGram
from word_sampler import WordSampler

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")


def _data_queue(corpus_data, *, n_threads, n_sentences, batch_size):
    data_loader = DataLoader(corpus_data, collate_fn=concat_collate, batch_size=n_sentences, num_workers=n_threads,
                             pin_memory=True, sampler=BlockRandomSampler(corpus_data))
    while True:
        for pos_u, pos_v, neg_v in data_loader:
            for i in range(pos_u.shape[0] // batch_size):
                pos_u_b = pos_u[i * batch_size: (i + 1) * batch_size].to(GPU)
                pos_v_b = pos_v[i * batch_size: (i + 1) * batch_size].to(GPU)
                neg_v_b = neg_v[i * batch_size: (i + 1) * batch_size].to(GPU)
                yield pos_u_b, pos_v_b, neg_v_b


class Trainer:
    def __init__(self, corpus_data_0, corpus_data_1, *, params, n_samples=10000000):
        self.skip_gram = [SkipGram(corpus_data_0.vocab_size + 1, params.emb_dim).to(GPU),
                          SkipGram(corpus_data_1.vocab_size + 1, params.emb_dim).to(GPU)]
        self.discriminator = Discriminator(params.emb_dim, n_layers=params.d_n_layers, n_units=params.d_n_units,
                                           drop_prob=params.d_drop_prob, drop_prob_input=params.d_drop_prob_input,
                                           leaky=params.d_leaky, batch_norm=params.d_bn).to(GPU)
        self.sg_optimizer, self.sg_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_sgd_linear(self.skip_gram[id].parameters(), params.n_steps,
                                                             lr=params.sg_lr)
            self.sg_optimizer.append(optimizer)
            self.sg_scheduler.append(scheduler)
        self.a_optimizer, self.a_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_sgd_linear(
                [{"params": self.skip_gram[id].u.parameters()}, {"params": self.skip_gram[id].v.parameters()}],
                params.n_steps, lr=params.a_lr)
            self.a_optimizer.append(optimizer)
            self.a_scheduler.append(scheduler)
        if params.d_optimizer == "SGD":
            self.d_optimizer, self.d_scheduler = optimizers.get_sgd_linear(self.discriminator.parameters(), params.n_steps,
                                                                           lr=params.d_lr, wd=params.d_wd)
        elif params.d_optimizer == "RMSProp":
            self.d_optimizer, self.d_scheduler = optimizers.get_rmsprop_linear(self.discriminator.parameters(), params.n_steps,
                                                                               lr=params.d_lr, wd=params.d_wd)
        else:
            raise Exception(f"Optimizer {params.d_optimizer} not found.")
        self.smooth = params.smooth
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        self.corpus_data_queue = [
            _data_queue(corpus_data_0, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.sg_bs),
            _data_queue(corpus_data_1, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.sg_bs)
        ]
        self.sampler = [WordSampler(corpus_data_0.dic, n_urns=n_samples, alpha=0.75),
                        WordSampler(corpus_data_1.dic, n_urns=n_samples, alpha=0.75)]
        self.d_bs = params.d_bs

    def skip_gram_step(self):
        losses = []
        for id in [0, 1]:
            self.sg_optimizer[id].zero_grad()
            pos_u_b, pos_v_b, neg_v_b = self.corpus_data_queue[id].__next__()
            pos_s, neg_s = self.skip_gram[id](pos_u_b, pos_v_b, neg_v_b)
            loss = SkipGram.loss_fn(pos_s, neg_s)
            loss.backward()
            self.sg_optimizer[id].step()
            losses.append(loss.item())
        return losses[0], losses[1]

    def get_adv_batch(self, *, reverse):
        batch = [torch.LongTensor([self.sampler[id].sample() for _ in range(self.d_bs)]).view(self.d_bs, 1).to(GPU)
                 for id in [0, 1]]
        x = [((self.skip_gram[id].u(batch[id]) + self.skip_gram[id].v(batch[id])) * 0.5).view(self.d_bs, -1)
             for id in [0, 1]]
        x = torch.cat(x, 0)
        y = torch.FloatTensor(self.d_bs * 2).to(GPU).uniform_(0.0, self.smooth)
        if reverse:
            y[: self.d_bs] = 1 - y[: self.d_bs]
        else:
            y[self.d_bs:] = 1 - y[self.d_bs:]
        return x, y

    def adversarial_step(self):
        for id in [0, 1]:
            self.a_optimizer[id].zero_grad()
        self.discriminator.eval()
        x, y = self.get_adv_batch(reverse=True)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        for id in [0, 1]:
            self.a_optimizer[id].step()
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

    def scheduler_step(self):
        for id in [0, 1]:
            self.sg_scheduler[id].step()
            self.a_scheduler[id].step()
        self.d_scheduler.step()


def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    parser = argparse.ArgumentParser(description="adversarial training")
    parser.add_argument("corpus_path_0", type=str, help="path of corpus 0")
    parser.add_argument("corpus_path_1", type=str, help="path of corpus 1")
    parser.add_argument("dic_path_0", type=str, help="path of dictionary 0")
    parser.add_argument("dic_path_1", type=str, help="path of dictionary 1")
    parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--n_threads", type=int, help="number of data loader threads")
    parser.add_argument("--sg_bs", type=int, help="batch size")
    parser.add_argument("--d_bs", type=int, help="batch size for the discriminator")
    parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
    parser.add_argument("--sg_lr", type=float, help="initial learning rate of skip-gram")
    parser.add_argument("--a_lr", type=float, help="max learning rate of adversarial training for embeddings")
    parser.add_argument("--d_lr", type=float, help="max learning rate of adversarial training for the discriminator")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_steps", type=int, help="number of iterations")
    parser.add_argument("--smooth", type=float, help="label smooth for adversarial training")
    parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--threshold", type=float, default=1e-4, help="sampling threshold")
    parser.add_argument("--checkpoint", type=bool, default=False, help="save a checkpoint after each epoch")

    parser.add_argument("--d_n_layers", type=int, help="number of hidden layers of the discriminator")
    parser.add_argument("--d_n_units", type=int, help="number of units per hidden layer of the discriminator")
    parser.add_argument("--d_drop_prob", type=float, help="dropout probability after each layer of the discriminator")
    parser.add_argument("--d_drop_prob_input", type=float, help="dropout probability of the input of the discriminator")
    parser.add_argument("--d_leaky", type=float, help="slope of leaky ReLU of the discriminator")
    parser.add_argument("--d_wd", type=float, help="weight decay of adversarial training for the discriminator")
    parser.add_argument("--d_bn", type=bool, help="turn on batch normalization for the discriminator or not")
    parser.add_argument("--d_optimizer", type=str, help="optimizer for the discriminator")

    parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
    parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")

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
    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(out_path, "log.txt"), use_incoming_socket=False)
    out_freq, checkpoint_freq = 500, params.n_steps // 10
    step, c, sg_loss, d_loss, a_loss = 0, 0, [0.0, 0.0], 0.0, 0.0
    for i in trange(params.n_steps):
        trainer.scheduler_step()
        l0, l1 = trainer.skip_gram_step()
        sg_loss[0] += l0
        sg_loss[1] += l1
        d_loss += trainer.discriminator_step()
        a_loss += trainer.adversarial_step()
        c += 1
        if c >= out_freq:
            vis.line(Y=torch.FloatTensor([trainer.sg_scheduler[0].factor * params.sg_lr]), X=torch.LongTensor([step]),
                     win="sg_lr", env=params.out_path, opts={"title": "sg_lr"}, update="append")
            vis.line(Y=torch.FloatTensor([trainer.a_scheduler[0].factor * params.a_lr]), X=torch.LongTensor([step]),
                     win="a_lr", env=params.out_path, opts={"title": "a_lr"}, update="append")
            vis.line(Y=torch.FloatTensor([trainer.d_scheduler.factor * params.d_lr]), X=torch.LongTensor([step]),
                     win="d_lr", env=params.out_path, opts={"title": "d_lr"}, update="append")
            vis.line(Y=torch.FloatTensor([sg_loss[0] / c]), X=torch.LongTensor([step]),
                     win="sg_loss_0", env=params.out_path, opts={"title": "sg_loss_0"}, update="append")
            vis.line(Y=torch.FloatTensor([sg_loss[1] / c]), X=torch.LongTensor([step]),
                     win="sg_loss_1", env=params.out_path, opts={"title": "sg_loss_1"}, update="append")
            vis.line(Y=torch.FloatTensor([d_loss / c]), X=torch.LongTensor([step]),
                     win="d_loss", env=params.out_path, opts={"title": "d_loss"}, update="append")
            vis.line(Y=torch.FloatTensor([a_loss / c]), X=torch.LongTensor([step]),
                     win="a_loss", env=params.out_path, opts={"title": "a_loss"}, update="append")
            c, sg_loss, d_loss, a_loss = 0, [0.0, 0.0], 0.0, 0.0
            step += 1
        if params.checkpoint and (i + 1) % checkpoint_freq == 0:
            torch.save({"skip_gram_0": trainer.skip_gram[0].state_dict(),
                        "skip_gram_1": trainer.skip_gram[1].state_dict(),
                        "discriminator": trainer.discriminator.state_dict()},
                       os.path.join(out_path, f"model-epoch{(i + 1) // checkpoint_freq}.pt"))
    torch.save({"skip_gram_0": trainer.skip_gram[0].state_dict(),
                "skip_gram_1": trainer.skip_gram[1].state_dict(),
                "discriminator": trainer.discriminator.state_dict()},
               os.path.join(out_path, f"model.pt"))
    print(params)


if __name__ == "__main__":
    main()
