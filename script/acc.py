import os
import sys
sys.path.append("../src")

import torch
import torch.nn.functional as F
from tqdm import trange
from vocab import Vocab
from dataset import StyleDataset, TemplateDataset
from torch.utils.data import DataLoader
from module.classifier import TextCNN

"""
python acc.py [corpus name] [hypothesis file]
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

test_dataset = TemplateDataset([hyp_file], vocab, borders=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_mask)
hits, total = 0, 0
# total_shifts = []

for _, x_tsf, y in test_loader:
    x_tsf, y = x_tsf.to(dev), y.to(dev)
    with torch.no_grad():
        pred_tsf = model(x_tsf)
    hits += torch.eq((pred_tsf > 0.5).long(), y).sum().item()
    total += y.size(0)
acc = hits / total
print("ACC: %.4f" % acc)