import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys 
import torch

from fio import split_lines
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig


"""
python gram.py [hypothesis file]
"""

hyp_file = sys.argv[1]
assert os.path.exists(hyp_file)


dev = torch.device("cuda")

tkz = AutoTokenizer.from_pretrained("bert-base-uncased", mirror="tuna")

bert = BertForSequenceClassification(BertConfig()).to(dev)
bert.load_state_dict(torch.load("../dump/eval_disc.pth"))
bert.eval()

class DiscDataset(Dataset):
    def __init__(self, src_tgt_pairs):
        super().__init__()
        self.samples = src_tgt_pairs 

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    src, tgt = zip(*batch)
    src_pt = tkz(list(src), return_tensors="pt", padding=True)["input_ids"].to(dev)
    tgt_pt = tkz(list(tgt), return_tensors="pt", padding=True)["input_ids"].to(dev)
    return src_pt, tgt_pt


data = [splitted[:2] for splitted in split_lines(hyp_file, "\t")]
train_loader = DataLoader(
    dataset=DiscDataset(data), batch_size=100, shuffle=False, collate_fn=collate_fn
)

cnt, hit = 0, 0
softmax = torch.nn.Softmax(dim=-1)
for src, tgt in train_loader:
    with torch.no_grad():
        src_preds = bert(input_ids=src)[0]
        tgt_preds = bert(input_ids=tgt)[0]
        src_norm, tgt_norm = softmax(src_preds)[:, 1], softmax(tgt_preds)[:, 1]
        hit += (tgt_norm >= src_norm).sum().item()
        cnt += src_norm.size(0)
print("Prop. pass the test: %.2f" % (100 * hit / cnt))