import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import visdom

import optimizers
from corpus_data import CorpusData, concat_collate
from skip_gram import SkipGram

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train skip-gram only")
    parser.add_argument("corpus_path", type=str, help="path of corpus")
    parser.add_argument("dic_path", type=str, help="path of dictionary")
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--n_threads", type=int, help="number of data loader threads")
    parser.add_argument("--bs", type=int, help="batch size")
    parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--log_path", type=str, default="log.txt", help="path of logs")
    params = parser.parse_args()

    corpus_data = CorpusData(params.corpus_path, params.dic_path, max_ws=params.max_ws, n_ns=params.n_ns)
    data_loader = DataLoader(corpus_data, collate_fn=concat_collate, batch_size=params.n_sentences,
                             num_workers=params.n_threads, pin_memory=True)
    model = SkipGram(corpus_data.vocab_size + 1, params.emb_dim).to(GPU)
    optimizer, scheduler = optimizers.get(model.parameters(), params.n_epochs * len(data_loader), lr=params.lr)

    vis = visdom.Visdom(port=params.vis_port, log_to_filename=params.log_path)
    for epoch in trange(params.n_epochs, desc="epoch"):
        for pos_u, pos_v, neg_v in tqdm(data_loader, desc=f"epoch {epoch}"):
            scheduler.step()
            vis.line(torch.FloatTensor(scheduler.factor * params.lr), win="lr", update="append")
            loss0, loss1 = 0, 0.0
            for i in range(pos_u.shape[0] // params.bs):
                optimizer.zero_grad()
                pos_u_b = pos_u[i * params.bs: (i + 1) * params.bs].to(GPU)
                pos_v_b = pos_v[i * params.bs: (i + 1) * params.bs].to(GPU)
                neg_v_b = neg_v[i * params.bs: (i + 1) * params.bs].to(GPU)
                pos_s, neg_s = model(pos_u_b, pos_v_b, neg_v_b)
                loss = SkipGram.loss_fn(pos_s, neg_s)
                loss0 += 1
                loss1 += loss.item()
                loss.backward()
                optimizer.step()
            vis.line(torch.FloatTensor([loss1 / loss0]), win="loss", update="append")
