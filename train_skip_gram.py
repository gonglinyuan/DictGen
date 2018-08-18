import argparse
import os
from datetime import datetime

import torch
import visdom
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import optimizers
from corpus_data import CorpusData, concat_collate
from skip_gram import SkipGram

GPU = torch.device("cuda:0")
CPU = torch.device("cpu")

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser(description="train skip-gram only")
    parser.add_argument("corpus_path", type=str, help="path of corpus")
    parser.add_argument("dic_path", type=str, help="path of dictionary")
    parser.add_argument("--out_path", type=str, default=timestamp, help="path of all outputs")
    parser.add_argument("--max_ws", type=int, help="max window size")
    parser.add_argument("--n_ns", type=int, help="number of negative samples")
    parser.add_argument("--n_threads", type=int, help="number of data loader threads")
    parser.add_argument("--bs", type=int, help="batch size")
    parser.add_argument("--n_sentences", type=int, help="number of sentences to load each time")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--emb_dim", type=int, help="dimensions of the embedding")
    parser.add_argument("--n_epochs", type=int, help="number of epochs")
    parser.add_argument("--vis_host", type=str, default="localhost", help="host name for Visdom")
    parser.add_argument("--vis_port", type=int, default=34029, help="port for Visdom")
    parser.add_argument("--threshold", type=float, default=1e-4, help="sampling threshold")
    parser.add_argument("--checkpoint", type=bool, default=False, help="save a checkpoint after each epoch")
    params = parser.parse_args()

    print(params)

    if not os.path.exists(params.out_path):
        os.mkdir(params.out_path)

    corpus_data = CorpusData(params.corpus_path, params.dic_path, max_ws=params.max_ws, n_ns=params.n_ns,
                             threshold=params.threshold)
    data_loader = DataLoader(corpus_data, collate_fn=concat_collate, batch_size=params.n_sentences,
                             num_workers=params.n_threads, pin_memory=True)
    model = SkipGram(corpus_data.vocab_size + 1, params.emb_dim).to(GPU)
    optimizer, scheduler = optimizers.get(model.parameters(), params.n_epochs * len(data_loader), lr=params.lr)

    vis = visdom.Visdom(server=f'http://{params.vis_host}', port=params.vis_port,
                        log_to_filename=os.path.join(params.out_path, "log.txt"))
    out_freq = (len(data_loader) + 99) // 100
    loss0, loss1, step = 0, 0.0, 0
    for epoch in trange(params.n_epochs, desc="epoch"):
        print(f"epoch {epoch} ; out_path = {params.out_path}")
        for pos_u, pos_v, neg_v in tqdm(data_loader, desc=f"epoch {epoch}"):
            scheduler.step()
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
            if loss0 >= out_freq:
                vis.line(Y=torch.FloatTensor([scheduler.factor * params.lr]), X=torch.LongTensor([step]),
                         win="lr", env=params.out_path, update="append")
                vis.line(Y=torch.FloatTensor([loss1 / loss0]), X=torch.LongTensor([step]),
                         win="loss", env=params.out_path, update="append")
                loss0, loss1 = 0, 0.0
                step += 1
        if params.checkpoint:
            torch.save(model.state_dict(), os.path.join(params.out_path, f"model-epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(params.out_path, f"model.pt"))
    print(params)
