import os
import sys
sys.path.append("../src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from vocab import Vocab, PLH_ID, PAD_ID
from dataset import StyleDataset, TemplateDataset
from nets.classifier import TextCNN
from bert_score import BERTScorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import ElectraTokenizer, ElectraForTokenClassification
from torch.utils.data import DataLoader, Dataset

"""
python prepare.py [corpus name]
"""

# arguments
ds = sys.argv[1]
print(f"dataset={ds}")

batch_size = 256
epochs = 10
p_noise = 0.15
dev = torch.device("cuda:0")

# prepare for BERT score (ALBERT-xlarge-v2)
print("Downloading BERT model...")
scorer = BERTScorer(model_type="albert-xlarge-v2", lang="en", rescale_with_baseline=True)
del scorer


# prepare for classification (Classifier)
print("Training <PH>-inserted classifier (TextCNN)...")

vocab_path = f"../dump/vocab_{ds}.bin"
vocab = Vocab.load(vocab_path)
clf_model = TextCNN(len(vocab)).to(dev)

train_files = [f"../data/{ds}/style.train.0", f"../data/{ds}/style.train.1"]
dev_files = [f"../data/{ds}/style.dev.0", f"../data/{ds}/style.dev.1"]

train_dataset = StyleDataset(train_files, vocab, max_len=None)
dev_dataset = StyleDataset(dev_files, vocab, max_len=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn_noise(p_noise, PLH_ID, "insert"))
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
                _, x, y = next(dataloader_iterator)
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

# # prepare for perplexity (Language Model from GPT-2)
# print("Training LM from GPT-2...")
# tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id
# lm_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(dev)
# optimizer = torch.optim.Adam(lm_model.parameters(), lr=1e-5)

# class LMDataset(Dataset):
#     def __init__(self, sentences):
#         super().__init__()
#         self.sentences = [[bos_id] + tokenizer.encode(s) + [eos_id] for s in sentences]
    
#     def __getitem__(self, index):
#         return self.sentences[index]
    
#     def __len__(self):
#         return len(self.sentences)

# def collate_fn(batch_samples):
#     max_len = max([len(s) for s in batch_samples])
#     batch = [s + [-1] * (max_len - len(s)) for s in batch_samples]
#     tensor = torch.tensor(batch, dtype=torch.long)
#     inputs, labels = tensor[:, :-1], tensor[:, 1:]
#     inputs[inputs == -1] = pad_id
#     return inputs.to(dev), labels.to(dev)

# def cal_ce_loss(logits, labels):
#     crit = nn.CrossEntropyLoss(ignore_index=-1)
#     return crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

# def fit(model, loader, optimizer, epoch):
#     model.train()
#     #===============================================
#     batch_iters = len(loader)
#     dataloader_iterator = iter(loader)
#     with trange(batch_iters) as t:
#         for _ in t:
#             t.set_description(f"Epoch {epoch}")
#             try:
#                 inp, labels = next(dataloader_iterator)
#             except StopIteration:
#                 break
#             logits = model(input_ids=inp)[0]
#             loss = cal_ce_loss(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             t.set_postfix(loss="%.2f" % loss.item())

# def evaluate(model, loader):
#     model.eval()
#     losses = []
#     with torch.no_grad():
#         for inp, labels in loader:
#             logits = model(input_ids=inp)[0]
#             loss = cal_ce_loss(logits, labels)
#             losses.append(loss.item())
#     return sum(losses)/len(losses)


# train_files = [f"../data/{ds}/style.train.0", f"../data/{ds}/style.train.1"]
# dev_files = [f"../data/{ds}/style.dev.0", f"../data/{ds}/style.dev.1"]
# train_sentences = [s.strip() for s in open(train_files[0], 'r')] + [s.strip() for s in open(train_files[1], 'r')]
# dev_sentences = [s.strip() for s in open(dev_files[0], 'r')] + [s.strip() for s in open(dev_files[1], 'r')]
# train_loader = DataLoader(
#     dataset=LMDataset(train_sentences), batch_size=64, shuffle=True, collate_fn=collate_fn
# )
# dev_loader = DataLoader(
#     dataset=LMDataset(dev_sentences), batch_size=64, shuffle=False, collate_fn=collate_fn
# )

# min_loss = evaluate(lm_model, dev_loader)

# for epoch in range(epochs // 2):
#     fit(lm_model, train_loader, optimizer, epoch)
#     eval_loss = evaluate(lm_model, dev_loader)
#     if eval_loss < min_loss:
#         # save
#         torch.save(lm_model.state_dict(), f"../dump/eval_lm_{ds}.pth")
#         print("Loss is optimized: %.2f -> %.2f" % (min_loss, eval_loss))
#         min_loss = eval_loss
#     else:
#         print("Done")
#         break

## download ELECTRA
# tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
# model = ElectraForTokenClassification.from_pretrained('google/electra-small-discriminator')