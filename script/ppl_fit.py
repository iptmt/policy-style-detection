import os
import sys
sys.path.append("../src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from vocab import Vocab, PLH_ID, PAD_ID
from bert_score import BERTScorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

"""
python ppl_fit.py [corpus name]
"""

# arguments
ds = sys.argv[1]
print(f"dataset={ds}")

batch_size = 256
epochs = 10
p_noise = 0.15
dev = torch.device("cuda:0")

# prepare for perplexity (Language Model from GPT-2)
print("Training LM from GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id
lm_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(dev)
optimizer = torch.optim.Adam(lm_model.parameters(), lr=1e-5)

class LMDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = [[bos_id] + tokenizer.encode(s) + [eos_id] for s in sentences]
    
    def __getitem__(self, index):
        return self.sentences[index]
    
    def __len__(self):
        return len(self.sentences)

def collate_fn(batch_samples):
    max_len = max([len(s) for s in batch_samples])
    batch = [s + [-1] * (max_len - len(s)) for s in batch_samples]
    tensor = torch.tensor(batch, dtype=torch.long)
    inputs, labels = tensor[:, :-1], tensor[:, 1:]
    inputs[inputs == -1] = pad_id
    return inputs.to(dev), labels.to(dev)

def cal_ce_loss(logits, labels):
    crit = nn.CrossEntropyLoss(ignore_index=-1)
    return crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

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
            logits = model(input_ids=inp)[0]
            loss = cal_ce_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss="%.2f" % loss.item())

def evaluate(model, loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for inp, labels in loader:
            logits = model(input_ids=inp)[0]
            loss = cal_ce_loss(logits, labels)
            losses.append(loss.item())
    return sum(losses)/len(losses)


train_files = [f"../data/{ds}/style.train.0", f"../data/{ds}/style.train.1"]
dev_files = [f"../data/{ds}/style.dev.0", f"../data/{ds}/style.dev.1"]
train_sentences = [s.strip() for s in open(train_files[0], 'r')] + [s.strip() for s in open(train_files[1], 'r')]
dev_sentences = [s.strip() for s in open(dev_files[0], 'r')] + [s.strip() for s in open(dev_files[1], 'r')]
train_loader = DataLoader(
    dataset=LMDataset(train_sentences), batch_size=64, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    dataset=LMDataset(dev_sentences), batch_size=64, shuffle=False, collate_fn=collate_fn
)

min_loss = evaluate(lm_model, dev_loader)

for epoch in range(epochs // 2):
    fit(lm_model, train_loader, optimizer, epoch)
    eval_loss = evaluate(lm_model, dev_loader)
    if eval_loss < min_loss:
        # save
        torch.save(lm_model.state_dict(), f"../dump/eval_lm_{ds}.pth")
        print("Loss is optimized: %.2f -> %.2f" % (min_loss, eval_loss))
        min_loss = eval_loss
    else:
        print("Done")
        break