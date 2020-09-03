"""
python attn_score.py [corpus name] [`fit' or `inf'] [gamma (optional)]

"""

import sys
sys.path.append("..")
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from nets.classifier import TextRNN
from vocab import Vocab
from dataset import StyleDataset
from trainer import MaskTrainer


ds = sys.argv[1]
md = sys.argv[2]
print(f"dataset: {ds}")
print(f"mode: {md}")

if md == "inf":
    gamma = float(sys.argv[3])
    print(f"gamma: {gamma}")

epochs = 10
batch_size = 512
dev = torch.device("cuda:0")

data_dir = f"../../data/{ds}/"
train_files = [data_dir + "style.train.0", data_dir + "style.train.1"]
dev_files = [data_dir + "style.dev.0", data_dir + "style.dev.1"]
test_files = [data_dir + "style.test.0", data_dir + "style.test.1"]
vocab_file = f"../../dump/vocab_{ds}.bin"
output_file = f"../../tmp/{ds}.test.mask.attn"

vocab = Vocab.load(vocab_file)
model = TextRNN(len(vocab)).to(dev)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def train_clf(dl, epoch):
    model.train()
    batch_iters = len(dl)
    dataloader_iterator = iter(dl)
    with trange(batch_iters) as t:
        for iters in t:
            t.set_description(f"Epoch {epoch}")
            try:
                x, y, m = next(dataloader_iterator)
            except StopIteration:
                break
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            pred = model(x, m)
            loss_cls = torch.nn.BCELoss()(pred, y.float())
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            t.set_postfix(step=f"{iters}/{batch_iters}", loss="%.2f" % loss_cls.item())

def eval_clf(dl):
    model.eval()
    hits, total = 0, 0
    for x, y, m in dl:
        x, y, m = x.to(dev), y.to(dev), m.to(dev)
        with torch.no_grad():
            pred = model(x, m)
        pred_lb = (pred > 0.5).long()
        hits += torch.eq(pred_lb, y).sum().item()
        total += x.size(0)
    return hits / total

def eval_inf(dl):
    model.eval()
    tensors = []
    for x, l, m in dl:
        x, m = x.to(dev), m.to(dev)
        norm_weights = model.get_weights(x, m) # B, L
        mask = ((norm_weights - gamma * norm_weights.mean(dim=1).unsqueeze(-1)) < 0).long()
        masked_x = mask * x
        results = list(zip(x.cpu().unbind(0), masked_x.cpu().unbind(0), m.cpu().unbind(0), l.unbind(0)))
        tensors += results
    return tensors


if md == "fit":
    train_dataset = StyleDataset(train_files, vocab, max_len=None)
    dev_dataset = StyleDataset(dev_files, vocab, max_len=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

    best_acc = eval_clf(dev_loader)
    for epoch in range(epochs):
        train_clf(train_loader, epoch)
        acc = eval_clf(dev_loader)
        print(f"Dev Acc: {acc}")
        if acc > best_acc:
            print(f"Update clf dump {int(acc*1e4)/1e4} <- {int(best_acc*1e4)/1e4}")
            torch.save(model.state_dict(), f"../../tmp/clf_{ds}.pth")
            best_acc = acc

if md == "inf":
    test_dataset = StyleDataset(test_files, vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

    model.load_state_dict(torch.load(f"../../tmp/clf_{ds}.pth"))

    results = eval_inf(test_loader)

    with open(output_file, "w+", encoding="utf-8") as f:
        for src, tgt, pad_mask, label in results:
            src = vocab.tensor_to_template(src, pad_mask)
            tgt = vocab.tensor_to_template(tgt, pad_mask)
            f.write(src + "\t" + tgt + "\t" + str(label.item()) + "\n")