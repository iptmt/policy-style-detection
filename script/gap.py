import os
import sys
sys.path.append("../src")
import torch
from vocab import Vocab, PAD_ID
from dataset import TemplateDataset
from torch.utils.data import DataLoader
from module.classifier import TextCNN

"""
python gap.py [corpus name] [hypothesis file]
"""

ds = sys.argv[1]
hyp_file = sys.argv[2]
assert os.path.exists(hyp_file)

vocab_path = f"../dump/vocab_{ds}.bin"

batch_size = 512
dev = torch.device("cuda:0")

vocab = Vocab.load(vocab_path)
model = TextCNN(len(vocab)).to(dev)
model.load_state_dict(torch.load(f"../dump/eval_clf_{ds}.pth"))
model.eval()

test_dataset = TemplateDataset([hyp_file], vocab)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_margin)
total_shifts, N_remain, N_total = [], 0, 0

for x, x_m, y in test_loader:
    x, x_m, y = x.to(dev), x_m.to(dev), y.to(dev)
    # pad_mask = (x != PAD_ID).long()
    with torch.no_grad():
        pred = model(x)
        pred_m = model(x_m)
    shifts = (1 - 2 * y.float()) * (pred_m - pred)
    total_shifts.append(shifts.cpu())
    N_remain += (x_m != PAD_ID).long().sum().item()
    N_total += (x != PAD_ID).long().sum().item()

avg_shift = torch.cat(total_shifts, 0).mean().item()
print("Prop.: %.4f, Average style shift: %.4f" % (1 - N_remain/N_total, avg_shift))