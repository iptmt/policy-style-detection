import sys
import torch
import random
import numpy as np
import torch

from vocab import Vocab
from dataset import TemplateDataset, StyleDataset
from torch.utils.data import DataLoader

from trainer import MLMTrainer
from nets.lm import MaskLM
from tool import create_logger
from vocab import PLH_ID

seed_num = 79
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

logger = create_logger("../log/", "rank.log")

data = sys.argv[1]
logger.info(f"data: {data}")
mode = sys.argv[2]
logger.info(f"mode: {mode}")

# parameters
#=============================================================#
epochs = 10
batch_size = 512
max_seq_len = None # no limit
dev = torch.device("cuda:0")
noise_p = 0.3

vocab_file = f"../dump/vocab_{data}.bin"
vb = Vocab.load(vocab_file)


if mode == "train":
    # load sources
    #=============================================================#
    train_files = [f"../data/{data}/style.train.0", f"../data/{data}/style.train.1"]
    dev_files = [f"../data/{data}/style.dev.0", f"../data/{data}/style.dev.1"]

    train_dataset = StyleDataset(train_files, vb, max_len=max_seq_len)
    dev_dataset = StyleDataset(dev_files, vb, max_len=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=StyleDataset.collate_fn_noise(noise_p, PLH_ID, "mask"))
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=StyleDataset.collate_fn_noise(noise_p, PLH_ID, "mask"))

    # create model
    #=============================================================#
    mlm = MaskLM(len(vb))

    # construct trainer
    #=============================================================#
    optimize_mlm = torch.optim.Adam(mlm.parameters(), lr=1e-4)
    model_trainer = MLMTrainer(mlm, dev, optimize_mlm)

    # training
    #=============================================================#
    best_loss = model_trainer.evaluate(dev_loader)
    for epoch in range(epochs):
        logger.info(f"Training MLM -- Epoch {epoch}")
        model_trainer.train(train_loader)
        loss = model_trainer.evaluate(dev_loader)
        logger.info(f"Dev. Loss: {loss}")
        if loss < best_loss:
            logger.info(f"Update MLM dump {int(loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}")
            torch.save(mlm.state_dict(), f"../dump/mlm_{data}.pth")
            best_loss = loss
        logger.info("=" * 50)

elif mode == "test":
    # load sources
    #=============================================================#
    train_files = [f"../tmp/{data}.train.mask"]
    dev_files = [f"../tmp/{data}.dev.mask"]
    test_files = [f"../tmp/{data}.test.mask"]

    train_dataset = TemplateDataset(train_files, vb, max_seq_len)
    dev_dataset= TemplateDataset(dev_files, vb, max_seq_len)
    test_dataset= TemplateDataset(test_files, vb, max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_rank)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_rank)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_rank)

    # load model
    #=============================================================#
    mlm = MaskLM(len(vb))
    mlm.load_state_dict(torch.load(f"../dump/mlm_{data}.pth"))

    # construct trainer
    #=============================================================#
    model_trainer = MLMTrainer(mlm, dev, None)

    # generate permutation
    #=============================================================#
    model_trainer.rank_words(train_loader, f"../tmp/{data}.train.rank", vb)
    model_trainer.rank_words(dev_loader, f"../tmp/{data}.dev.rank", vb)
    model_trainer.rank_words(test_loader, f"../tmp/{data}.test.rank", vb)