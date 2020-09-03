import os
import sys
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

"""
python ppl.py [corpus name] [hypothesis file]
"""

ds = sys.argv[1]
hyp_file = sys.argv[2]
assert os.path.exists(hyp_file)

batch_size = 128
dev = torch.device("cuda:0")

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id

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


lm_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(dev)
lm_model.load_state_dict(torch.load(f"../dump/eval_lm_{ds}.pth"))
lm_model.eval()

sentences = [l.strip().split("\t")[1].strip() for l in open(hyp_file, 'r')]
test_loader = DataLoader(
    dataset=LMDataset(sentences), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

losses, counts = [], []
for inp, labels in test_loader:
    with torch.no_grad():
        logits = lm_model(input_ids=inp)[0]
    mask = (labels >= 0).float() # B, L
    loss_bat_avg = cal_ce_loss(logits, labels)

    losses.append(loss_bat_avg.item())
    counts.append(mask.sum().item())

losses = torch.tensor(losses)
counts = torch.tensor(counts)

loss_avg = ((counts / counts.sum()) * losses).sum()
ppl = torch.exp(loss_avg).item()
print("PPL: %.2f" % ppl)