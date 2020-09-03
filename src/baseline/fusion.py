"""
python fusion.py [corpus name] [threshold]

"""

import sys
sys.path.append("..")
import math
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from nets.classifier import TextRNN
from vocab import Vocab, PLH_ID
from dataset import StyleDataset
from trainer import MaskTrainer


ds = sys.argv[1]
ts = float(sys.argv[2])
print(f"dataset: {ds}")
print(f"threshold: {ts}")

epochs = 10
batch_size = 512
dev = torch.device("cuda:0")

data_dir = f"../../data/{ds}/"
train_files = [data_dir + "style.train.0", data_dir + "style.train.1"]
dev_files = [data_dir + "style.dev.0", data_dir + "style.dev.1"]
test_0 = f"../../data/{ds}/style.test.0"
test_1 = f"../../data/{ds}/style.test.1"
# test_files = [data_dir + "style.test.0", data_dir + "style.test.1"]
gram_file = f"../../tmp/grams_{ds}.bin"
vocab_file = f"../../dump/vocab_{ds}.bin"
output_file = f"../../tmp/{ds}.test.mask.fuse"

vocab = Vocab.load(vocab_file)
model = TextRNN(len(vocab)).to(dev)

test_dataset = StyleDataset([test_0, test_1], vocab)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

model.load_state_dict(torch.load(f"../../tmp/clf_{ds}.pth"))
model.eval()

weights = []
with torch.no_grad():
    for x, l, m in test_loader:
        x, m = x.to(dev), m.to(dev)
        norm_weights = model.get_weights(x, m) # B, L
        weights += norm_weights.cpu().unbind(0)
weights = [w.numpy() for w in weights]


def mask_sentences(word_lv_sents, src_dict, tgt_dict, threshold, v_smooth=1.0, mask_token=PLH_ID):
    masked_sents = []
    with tqdm(total=len(word_lv_sents)) as bar:
        for sent, weight in zip(word_lv_sents, weights):
            weight = weight[:len(sent)]
            sent = np.array(sent)
            inds = [0.] * len(sent)
            for idx in range(len(sent)):
                for i in range(1, 5):
                    if idx + i <= len(sent):
                        gram = tuple(sent[idx: idx + i])
                        src_score = src_dict[gram] if gram in src_dict else 0
                        tgt_score = tgt_dict[gram] if gram in tgt_dict else 0
                        salience = (src_score + v_smooth) / (tgt_score + v_smooth)
                        for j in range(idx, idx + i):
                            if inds[j] < salience:
                                inds[j] = salience
            inds = ((np.array(inds) * weight) > threshold).astype(np.long)
            masked_sent = sent * (1 - inds) + PLH_ID * inds
            masked_sents.append(masked_sent.tolist())
            bar.update()
    return masked_sents

grams_0, grams_1 = pickle.load(open(gram_file, "rb"))
corpus_0 = [vocab.tokens_to_ids(line.split()) for line in open(test_0, "r", encoding="utf-8")]
corpus_1 = [vocab.tokens_to_ids(line.split()) for line in open(test_1, "r", encoding="utf-8")]
masked_0 = mask_sentences(corpus_0, grams_0, grams_1, ts)
masked_1 = mask_sentences(corpus_1, grams_1, grams_0, ts)
labels_0 = ["0" for _ in range(len(corpus_0))]
labels_1 = ["1" for _ in range(len(corpus_1))]

with open(output_file, "w+", encoding='utf-8') as f:
    for obj in (zip(corpus_0, masked_0, labels_0), zip(corpus_1, masked_1, labels_1)):
        for src, masked, label in obj:
            src = " ".join(vocab.ids_to_tokens(src))
            masked = " ".join(vocab.ids_to_tokens(masked))
            f.write(src + "\t" + masked + "\t" + label + "\n")