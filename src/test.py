import torch
from vocab import Vocab, PAD_ID, PLH_ID
from nets.classifier import TextCNN

dev = torch.device("cuda:0")

vocab = Vocab.load("../dump/vocab_yelp.bin")

model = TextCNN(len(vocab)).to(dev)
model.load_state_dict(torch.load("../dump/clf_yelp.pth"))
model.eval()

while True:
    query = input("Input a sentence: ")
    tokens = query.strip().split()
    ids = vocab.tokens_to_ids(tokens)
    # src = torch.tensor(ids).long().to(dev)
    ids = list(map(lambda x: x if x != PLH_ID else PAD_ID, ids))
    tgt = torch.tensor(ids).long().to(dev)

    with torch.no_grad():
        res = model(tgt.unsqueeze(0))
        print(res.item())