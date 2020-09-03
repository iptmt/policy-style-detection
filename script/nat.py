import os
import sys
import torch

from fio import read
from transformers import ElectraTokenizer, ElectraForPreTraining



"""
python ppl.py [hypothesis file]
"""

hyp_file = sys.argv[1]
assert os.path.exists(hyp_file)

device = torch.device("cuda:0")

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = ElectraForPreTraining.from_pretrained('google/electra-small-discriminator')
model.to(device)
model.eval()

sigmoid = torch.nn.Sigmoid()
pad_id = tokenizer.pad_token_id

src_tsf_paris = [line.split("\t")[:2] for line in read(hyp_file)]
src_tsf_paris = [(tokenizer.encode(src), tokenizer.encode(tsf)) for src, tsf in src_tsf_paris]

def format_samples(pairs):
    src_bat, tsf_bat = zip(*pairs)
    max_l_src, max_l_tsf = max([len(src) for src in src_bat]), max([len(tsf) for tsf in tsf_bat])
    src_bat = [src + [pad_id] * (max_l_src - len(src)) for src in src_bat]
    tsf_bat = [tsf + [pad_id] * (max_l_tsf - len(tsf)) for tsf in tsf_bat]
    return torch.tensor(src_bat).long().to(device), torch.tensor(tsf_bat).long().to(device)

def chunk(pairs, batch_size=256):
    point, chunks = 0, []
    while point < len(pairs):
        chunks.append(pairs[point: point + batch_size])
        point += batch_size
    return chunks
 
for batch_samples in chunk(src_tsf_paris):
    src_bat, tsf_bat = format_samples(batch_samples)
    src_mask = (src_bat == pad_id).float().unsqueeze(-1)
    tsf_mask = (tsf_bat == pad_id).float().unsqueeze(-1)
    scores_src = sigmoid(model(input_ids=src_bat)[0])
    scores_tsf = sigmoid(model(input_ids=tsf_bat)[0])