import sys

import torch

from src.dictionary import Dictionary

CPU = torch.device("cpu")


def convert_dic(dic, lang):
    id2word, word2id = {}, {}
    for i in range(len(dic)):
        id2word[i] = dic[i][0]
        word2id[dic[i][0]] = i
    return Dictionary(id2word, word2id, lang)


if __name__ == '__main__':
    model_path, dic_src_path, dic_tgt_path = sys.argv[1], sys.argv[2], sys.argv[3]
    src_lang, tgt_lang = sys.argv[4], sys.argv[5]
    out_src_path, out_tgt_path = sys.argv[6], sys.argv[7]
    model = torch.load(model_path)
    src_emb = ((model["skip_gram_0"]["u.weight"].to(CPU) + model["skip_gram_0"]["v.weight"].to(CPU)) * 0.5)[:-1]
    tgt_emb = ((model["skip_gram_1"]["u.weight"].to(CPU) + model["skip_gram_1"]["v.weight"].to(CPU)) * 0.5)[:-1]
    dic_src, dic_tgt = torch.load(dic_src_path), torch.load(dic_tgt_path)
    src_dic = convert_dic(dic_src, src_lang)
    tgt_dic = convert_dic(dic_tgt, tgt_lang)
    torch.save({"dico": src_dic, "vectors": src_emb}, out_src_path)
    torch.save({"dico": tgt_dic, "vectors": tgt_emb}, out_tgt_path)
