
import os
import sys
sys.path.append("../src")

import torch
from tqdm import trange
from vocab import Vocab, PLH_ID, PAD_ID
from dataset import StyleDataset, TemplateDataset
from module.classifier import TextCNN
from torch.utils.data import DataLoader

"""
python acc_fit.py [corpus name]
"""

# arguments
ds = sys.argv[1]
print(f"dataset={ds}")

batch_size = 256
epochs = 10
p_noise = 0.15
dev = torch.device("cuda:0")

print("Training classifier (TextCNN)...")

vocab_path = f"../dump/vocab_{ds}.bin"
vocab = Vocab.load(vocab_path)
clf_model = TextCNN(len(vocab)).to(dev)

train_files = [f"../data/{ds}/style.train.0", f"../data/{ds}/style.train.1"]
dev_files = [f"../data/{ds}/style.dev.0", f"../data/{ds}/style.dev.1"]

train_dataset = StyleDataset(train_files, vocab, max_len=None)
dev_dataset = StyleDataset(dev_files, vocab, max_len=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-3)

def train_clf(model, dl, optimizer, epoch):
    model.train()
    batch_iters = len(dl)
    dataloader_iterator = iter(dl)
    with trange(batch_iters) as t:
        for iters in t:
            t.set_description(f"Epoch {epoch}")
            try:
                x, y, _ = next(dataloader_iterator)
            except StopIteration:
                break
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            loss_cls = torch.nn.BCELoss()(pred, y.float())
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            t.set_postfix(step=f"{iters}/{batch_iters}", loss="%.2f" % loss_cls.item())

def eval_clf(model, dl):
    model.eval()
    hits, total = 0, 0
    for x, y, _ in dl:
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            pred = model(x)
        pred_lb = (pred > 0.5).long()
        hits += torch.eq(pred_lb, y).sum().item()
        total += x.size(0)
    return hits / total

best_acc = eval_clf(clf_model, dev_loader)
for epoch in range(epochs):
    train_clf(clf_model, train_loader, optimizer, epoch)
    acc = eval_clf(clf_model, dev_loader)
    print(f"Dev Acc: {acc}")
    if acc > best_acc:
        print(f"Update clf dump {int(acc*1e4)/1e4} <- {int(best_acc*1e4)/1e4}")
        torch.save(clf_model.state_dict(), f"../dump/eval_clf_{ds}.pth")
        best_acc = acc
del clf_model, train_loader, dev_loader