import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optimizers
from corpus_data import concat_collate, BlockRandomSampler
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
    def __init__(self, corpus_data_0, corpus_data_1, *, params):
        self.skip_gram = [SkipGram(corpus_data_0.vocab_size + 1, params.emb_dim).to(GPU),
                          SkipGram(corpus_data_1.vocab_size + 1, params.emb_dim).to(GPU)]
        self.discriminator = Discriminator(params.emb_dim, n_layers=params.d_n_layers, n_units=params.d_n_units,
                                           drop_prob=params.d_drop_prob, drop_prob_input=params.d_drop_prob_input,
                                           leaky=params.d_leaky)
        self.sg_optimizer, self.sg_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_skip_gram(self.skip_gram[id].parameters(), params.n_steps,
                                                            lr=params.sg_lr)
            self.sg_optimizer.append(optimizer)
            self.sg_scheduler.append(scheduler)
        self.a_optimizer, self.a_scheduler = [], []
        for id in [0, 1]:
            optimizer, scheduler = optimizers.get_adv(
                [{"params": self.skip_gram[id].u.parameters()}, {"params": self.skip_gram[id].v.parameters()}],
                params.n_steps, lr=params.a_lr, apex=params.apex)
            self.a_optimizer.append(optimizer)
            self.a_scheduler.append(scheduler)
        self.d_optimizer, self.d_scheduler = optimizers.get_adv(self.discriminator.parameters(), params.n_steps,
                                                                lr=params.a_lr, apex=params.apex)
        self.smooth = params.smooth
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="elementwise_mean")
        self.corpus_data_queue = [
            _data_queue(corpus_data_0, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.bs),
            _data_queue(corpus_data_1, n_threads=(params.n_threads + 1) // 2, n_sentences=params.n_sentences,
                        batch_size=params.bs)
        ]
        self.sampler = [WordSampler(corpus_data_0.dic, n_urns=params.n_negatives, alpha=0.5), WordSampler(corpus_data_1.dic, n_urns=params.n_negatives, alpha=0.5)]
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
            losses[id] = loss.item()
        return losses[0], losses[1]

    def get_adv_batch(self, *, reverse):
        batch = [torch.LongTensor([self.sampler[id].sample() for _ in range(self.d_bs)]).view(self.d_bs, 1)
                 for id in [0, 1]]
        x = [((self.skip_gram[id].u(batch[id]) + self.skip_gram[id].v(batch[id])) * 0.5).view(self.d_bs, -1)
             for id in [0, 1]]
        x = torch.cat(x, 0)
        y = torch.FloatTensor(self.d_bs * 2)
        if reverse:
            y[: self.d_bs] = 1 - self.smooth
            y[self.d_bs:] = self.smooth
        else:
            y[: self.d_bs] = self.smooth
            y[self.d_bs:] = 1 - self.smooth
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
            x, y = self.get_adv_batch(reverse=True)
        y_hat = self.discriminator(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

# if __name__ == "__main__":
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#
#     parser = argparse.ArgumentParser(description="train skip-gram only")
#     parser.add_argument("corpus_path", type=str, help="path of corpus")
#     parser.add_argument("dic_path", type=str, help="path of dictionary")
#     parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
#     parser.add_argument("--max_ws", type=int, help="max window size")
#     parser.add_argument("--n_ns", type=int, help="number of negative samples")
#     parser.add_argument("--n_threads", type=int, help="number of data loader threads")
#     parser.add_argument("--bs", type=int, help="batch size")
#     parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
#     parser.add_argument("--lr", type=float, help="initial learning rate")
#     parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
#     parser.add_argument("--n_epochs", type=int, help="number of epochs")
#     parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
#     parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
#     parser.add_argument("--threshold", type=float, default=1e-4, help="sampling threshold")
#     parser.add_argument("--checkpoint", type=bool, default=False, help="save a checkpoint after each epoch")
#
#     parser.add_argument("--dataDir", type=str, default=".", help="path for data (Philly only)")
#     parser.add_argument("--modelDir", type=str, default=".", help="path for outputs (Philly only)")
#     params = parser.parse_args()
#
#     print(params)
#
#     out_path = os.path.join(params.modelDir, params.out_path)
#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#
#     corpus_data = CorpusData(os.path.join(params.dataDir, params.corpus_path),
#                              os.path.join(params.dataDir, params.dic_path),
#                              max_ws=params.max_ws, n_ns=params.n_ns, threshold=params.threshold)
#     data_loader = DataLoader(corpus_data, collate_fn=concat_collate, batch_size=params.n_sentences,
#                              num_workers=params.n_threads, pin_memory=True, sampler=BlockRandomSampler(corpus_data))
#     model = SkipGram(corpus_data.vocab_size + 1, params.emb_dim).to(GPU)
#     optimizer, scheduler = optimizers.get(model.parameters(), params.n_epochs * len(data_loader), lr=params.lr)
#     vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
#                         log_to_filename=os.path.join(out_path, "log.txt"))
#     out_freq = (len(data_loader) + 99) // 100
#     loss0, loss1, step, mini_step = 0, 0.0, 0, 0
#     for epoch in trange(params.n_epochs, desc="epoch"):
#         print(f"epoch {epoch} ; out_path = {out_path}")
#         for pos_u, pos_v, neg_v in tqdm(data_loader, desc=f"epoch {epoch}"):
#             scheduler.step()
#             for i in range(pos_u.shape[0] // params.bs):
#                 optimizer.zero_grad()
#                 pos_u_b = pos_u[i * params.bs: (i + 1) * params.bs].to(GPU)
#                 pos_v_b = pos_v[i * params.bs: (i + 1) * params.bs].to(GPU)
#                 neg_v_b = neg_v[i * params.bs: (i + 1) * params.bs].to(GPU)
#                 pos_s, neg_s = model(pos_u_b, pos_v_b, neg_v_b)
#                 loss = SkipGram.loss_fn(pos_s, neg_s)
#                 loss0 += 1
#                 loss1 += loss.item()
#                 loss.backward()
#                 optimizer.step()
#             left_portion = (pos_u.shape[0] % params.bs) / params.bs
#             if left_portion > 0.01:
#                 # If the samples left are not negligible
#                 optimizer.zero_grad()
#                 pos_u_b = pos_u[(pos_u.shape[0] // params.bs) * params.bs:].to(GPU)
#                 pos_v_b = pos_v[(pos_u.shape[0] // params.bs) * params.bs:].to(GPU)
#                 neg_v_b = neg_v[(pos_u.shape[0] // params.bs) * params.bs:].to(GPU)
#                 pos_s, neg_s = model(pos_u_b, pos_v_b, neg_v_b)
#                 loss = SkipGram.loss_fn(pos_s, neg_s) * left_portion
#                 loss0 += left_portion
#                 loss1 += loss.item()
#                 loss.backward()
#                 optimizer.step()
#             mini_step += 1
#             if mini_step >= out_freq:
#                 vis.line(Y=torch.FloatTensor([scheduler.factor * params.lr]), X=torch.LongTensor([step]),
#                          win="lr", env=params.out_path, update="append")
#                 vis.line(Y=torch.FloatTensor([loss1 / loss0]), X=torch.LongTensor([step]),
#                          win="loss", env=params.out_path, update="append")
#                 loss0, loss1, mini_step = 0, 0.0, 0
#                 step += 1
#         if params.checkpoint:
#             torch.save(model.state_dict(), os.path.join(out_path, f"model-epoch{epoch}.pt"))
#
#     torch.save(model.state_dict(), os.path.join(out_path, f"model.pt"))
#     print(params)
