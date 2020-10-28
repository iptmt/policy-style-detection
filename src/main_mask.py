import sys
import torch
import random
import numpy as np

from vocab import Vocab
from dataset import StyleDataset
from torch.utils.data import DataLoader

from module.classifier import TextCNN
from module.masker import Masker
from trainer import MaskTrainer
from tool import create_logger
from vocab import PAD_ID

"""
python main_mask.py [corpus name] [`train' or `test']
"""

seed_num = 110
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

logger = create_logger("../log/", "mask.log")

data = sys.argv[1]
logger.info(f"data: {data}")
mode = sys.argv[2]
logger.info(f"mode: {mode}")

# parameters
#=============================================================#
epochs_clf = 10
epochs_masker = 40
batch_size = 512
max_seq_len = None # no limit
noise_p = 0.2
delta = 0.65

rollouts = 8
gamma = 0.90

dev = torch.device("cuda:0")
vocab_file = f"../dump/vocab_{data}.bin"
train_files = [f"../data/{data}/style.train.0", f"../data/{data}/style.train.1"]
dev_files = [f"../data/{data}/style.dev.0", f"../data/{data}/style.dev.1"]
test_files = [f"../data/{data}/style.test.0", f"../data/{data}/style.test.1"]
#=============================================================#

# load sources
#=============================================================#
vb = Vocab.load(vocab_file)

# create model
#=============================================================#
clf = TextCNN(len(vb))
masker = Masker(len(vb), delta)
#=============================================================#

if mode == "train":
    # load resources
    #=============================================================#
    train_dataset = StyleDataset(train_files, vb, max_len=None)
    dev_dataset = StyleDataset(dev_files, vb, max_len=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn_noise(noise_p, PAD_ID, "random"))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)
    #=============================================================#

    # construct trainer
    #=============================================================#
    optimize_clf = torch.optim.Adam(clf.parameters(), lr=1e-3)
    optimize_masker = torch.optim.Adam(masker.parameters(), lr=1e-4)
    model_trainer = MaskTrainer(masker, clf, dev, rollouts, gamma, optimize_masker, optimize_clf)
    #=============================================================#

    # pre-training
    #=============================================================#
    best_acc = model_trainer.eval_clf(dev_loader)
    for epoch in range(epochs_clf):
        logger.info(f"Pre-training classifier -- Epoch {epoch}")
        model_trainer.train_clf(train_loader)
        acc = model_trainer.eval_clf(dev_loader)
        logger.info(f"Dev Acc: {acc}")
        if acc > best_acc:
            logger.info(f"Update clf dump {int(acc*1e4)/1e4} <- {int(best_acc*1e4)/1e4}")
            torch.save(clf.state_dict(), f"../dump/clf_{data}.pth")
            best_acc = acc
        logger.info("=" * 50)
    clf.load_state_dict(torch.load(f"../dump/clf_{data}.pth"))

    del train_dataset, dev_dataset, train_loader, dev_loader

    #=============================================================#

    # load resources
    #=============================================================#
    train_dataset = StyleDataset(train_files, vb, max_len=max_seq_len)
    dev_dataset = StyleDataset(dev_files, vb, max_len=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

    # Training
    #=============================================================#
    best_r = model_trainer.evaluate(dev_loader)
    for epoch in range(epochs_masker):
        logger.info(f"Training masker -- Epoch {epoch}")
        model_trainer.train(train_loader)
        r = model_trainer.evaluate(dev_loader)
        if r > best_r:
            logger.info(f"Update masker dump {int(r*1e4)/1e4} <- {int(best_r*1e4)/1e4}")
            torch.save(masker.state_dict(), f"../dump/masker_{data}_{delta}.pth")
            best_r = r
        logger.info("=" * 50)
    #=============================================================#
elif mode == "inf":
    train_dataset = StyleDataset(train_files, vb)
    dev_dataset = StyleDataset(dev_files, vb)
    test_dataset = StyleDataset(test_files, vb)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn)

    masker.load_state_dict(torch.load(f"../dump/masker_{data}_{delta}.pth"))
    clf.load_state_dict(torch.load(f"../dump/clf_{data}.pth"))
    model_trainer = MaskTrainer(masker, clf, dev, rollouts, gamma, None, None)

    # Inference
    #=============================================================#
    model_trainer.inference(train_loader, f"../tmp/{data}.train.mask", vb)
    model_trainer.inference(dev_loader, f"../tmp/{data}.dev.mask", vb)
    model_trainer.inference(test_loader, f"../tmp/{data}.test.mask", vb)
    #=============================================================#
