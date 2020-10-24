import os
import sys
import csv
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch

from fio import read_tsv
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers import BertForSequenceClassification, AutoTokenizer


"""
python gram_fit.py
"""

batch_size = 64
lr = 2e-5
epochs = 5
dev = torch.device("cuda")
train_data = "../data/cola/in_domain_train.tsv"
dev_data = "../data/cola/out_of_domain_dev.tsv"
# dev_data = "../data/cola/in_domain_dev.tsv"

#tkz = AutoTokenizer.from_pretrained("roberta-base", mirror="tuna")
#
#roberta = RobertaForSequenceClassification.from_pretrained("roberta-base", mirror="tuna").to(dev)

tkz = AutoTokenizer.from_pretrained("bert-base-uncased", mirror="tuna")

bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", mirror="tuna").to(dev)
bert.train()

optimizer = torch.optim.AdamW(bert.parameters(), lr=lr)

class ClfDataset(Dataset):
    def __init__(self, sentence_label_pairs):
        super().__init__()
        self.samples  = sentence_label_pairs

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def collate_fn(batch_samples):
    texts, labels = zip(*batch_samples)
    texts_pt = tkz(list(texts), return_tensors="pt", padding=True)["input_ids"].to(dev)
    labels_pt = torch.tensor(list(labels)).long().to(dev)
    return texts_pt, labels_pt

def fit(model, loader, optimizer, epoch):
    model.train()
    #===============================================
    batch_iters = len(loader)
    dataloader_iterator = iter(loader)
    with trange(batch_iters) as t:
        for _ in t:
            t.set_description(f"Epoch {epoch}")
            try:
                inp, labels = next(dataloader_iterator)
            except StopIteration:
                break
            loss = model(input_ids=inp, labels=labels)[0]
            # loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss="%.2f" % loss.item())

def evaluate(model, loader):
    model.eval()
    hit, cnt = 0, 0
    with torch.no_grad():
        for inp, labels in loader:
            logits = model(input_ids=inp)[0]
            preds = logits.argmax(-1)
            cnt += labels.size(0)
            hit += torch.eq(preds, labels).sum().item()
    return hit / cnt

train_pairs = [(row[3], int(row[1])) for row in read_tsv(train_data)]
dev_pairs = [(row[3], int(row[1])) for row in read_tsv(dev_data)]

train_loader = DataLoader(
    dataset=ClfDataset(train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

dev_loader = DataLoader(
    dataset=ClfDataset(dev_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

best_acc = evaluate(bert, dev_loader)
for epoch in range(epochs):
    fit(bert, train_loader, optimizer, epoch)
    acc = evaluate(bert, dev_loader)
    if acc > best_acc:
        torch.save(bert.state_dict(), f"../dump/eval_disc.pth")
        print("Acc: %.4f -> %.4f" % (best_acc, acc))
        best_acc = acc
