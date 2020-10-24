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

for src, tgt in train_loader:
    print(src.shape, tgt.shape)