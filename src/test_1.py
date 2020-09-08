import torch
from torch.utils.data import DataLoader
from dataset import StyleDataset
from vocab import Vocab, PAD_ID, PLH_ID
from nets.classifier import TextCNN

dev = torch.device("cuda:0")

vocab = Vocab.load("../dump/vocab_yelp.bin")

model = TextCNN(len(vocab)).to(dev)
model.load_state_dict(torch.load("../dump/clf_yelp.pth"))


dev_files = ["../data/yelp/style.train.1"]
dev_dataset = StyleDataset(dev_files, vocab, max_len=None)
dev_loader = DataLoader(dev_dataset, batch_size=512, shuffle=False, collate_fn=StyleDataset.collate_fn)

idx = 0
cnt = 0
model.eval()
for x, y, _ in dev_loader:
    x, y = x.to(dev), y.to(dev)
    with torch.no_grad():
        pred = (model(x) > 0.5).long()
        inds = torch.eq(pred, y)
    for id_, x in enumerate(inds.cpu().numpy()):
        if x == 0:
            print(idx + id_ + 1)
            cnt += 1
    idx += y.size(0)
    if idx > 10000:
        break
    
print(cnt)