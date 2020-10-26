import os
import sys
import torch
import random
import numpy as np

from vocab import Vocab
from dataset import InsertLMDataset, TemplateDataset
from torch.utils.data import DataLoader

from trainer import InsertLMTrainer, MLMTrainer
from nets.lm import InsertLM, MaskLM
from tool import create_logger
from data_util import noise_text
from vocab import PLH

seed_num = 79
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

logger = create_logger("../log/", "insert.log")

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

train_files = [f"../tmp/{data}.train.rank"]
dev_files = [f"../tmp/{data}.dev.rank"]
test_files = [f"../tmp/{data}.test.mask"]


if mode == "pretrain":
    # generate randomly masked samples
    #=============================================================#
    with open(f"../data/{data}/style.train.0", "r") as f:
        ns = [line.strip().split() for line in f]
    with open(f"../data/{data}/style.train.1", "r") as f:
        ps = [line.strip().split() for line in f]
    with open(f"../data/{data}/style.dev.0", "r") as f:
        ns_ = [line.strip().split() for line in f]
    with open(f"../data/{data}/style.dev.1", "r") as f:
        ps_ = [line.strip().split() for line in f]

    mlm = MaskLM(len(vb))
    mlm.load_state_dict(torch.load(f"../dump/mlm_{data}.pth"))
    mlm_trainer = MLMTrainer(mlm, dev, None)

    ilm = InsertLM(len(vb))
    optimize_ilm = torch.optim.Adam(ilm.parameters(), lr=1e-4)
    model_trainer = InsertLMTrainer(ilm, dev, optimize_ilm)
    best_loss = 100.

    for epoch in range(epochs):
        nps = [(s, noise_text(s, noise_p, PLH), 1) for s in ps]
        nns = [(s, noise_text(s, noise_p, PLH), 0) for s in ns]
        nps_ = [(s, noise_text(s, noise_p, PLH), 1) for s in ps_]
        nns_ = [(s, noise_text(s, noise_p, PLH), 0) for s in ns_]

        with open(f"../tmp/{data}.train.mask_", "w+") as f:
            for s, noise_s, label in (nps + nns):
                f.write(" ".join(s) + "\t" + " ".join(noise_s) + "\t" + str(label) + "\n")
        with open(f"../tmp/{data}.dev.mask_", "w+") as f:
            for s, noise_s, label in (nps_ + nns_):
                f.write(" ".join(s) + "\t" + " ".join(noise_s) + "\t" + str(label) + "\n")

        # rank
        #=============================================================#
        trl = DataLoader(TemplateDataset([f"../tmp/{data}.train.mask_"], vb, max_seq_len), batch_size=batch_size, shuffle=True, collate_fn=TemplateDataset.collate_fn_rank)
        devl = DataLoader(TemplateDataset([f"../tmp/{data}.dev.mask_"], vb, max_seq_len), batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_rank)
        mlm_trainer.rank_words(trl, f"../tmp/{data}.train.rank_", vb)
        mlm_trainer.rank_words(devl, f"../tmp/{data}.dev.rank_", vb)

        # pretrain
        #=============================================================#
        train_dataset = InsertLMDataset([f"../tmp/{data}.train.rank_"], vb, max_len=max_seq_len)
        dev_dataset = InsertLMDataset([f"../tmp/{data}.dev.rank_"], vb, max_len=max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=InsertLMDataset.collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=InsertLMDataset.collate_fn)

        logger.info(f"Pre-training InsertLM -- Epoch {epoch}")
        model_trainer.train(train_loader)
        loss = model_trainer.evaluate(dev_loader)
        logger.info(f"Dev. Loss: {loss}")
        if loss < best_loss:
            logger.info(f"Update InsertLM dump {int(loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}")
            torch.save(ilm.state_dict(), f"../dump/ilm_{data}_pretrain.pth")
            best_loss = loss
        logger.info("=" * 50)
        os.remove(f"../tmp/{data}.train.mask_")
        os.remove(f"../tmp/{data}.dev.mask_")
        os.remove(f"../tmp/{data}.train.rank_")
        os.remove(f"../tmp/{data}.dev.rank_")


elif mode == "train":
    # load sources
    #=============================================================#
    train_dataset = InsertLMDataset(train_files, vb, max_len=max_seq_len)
    dev_dataset = InsertLMDataset(dev_files, vb, max_len=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=InsertLMDataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=InsertLMDataset.collate_fn)

    # create model
    #=============================================================#
    ilm = InsertLM(len(vb))
    ilm.load_state_dict(torch.load(f"../dump/ilm_{data}_pretrain.pth"))

    # construct trainer
    #=============================================================#
    optimize_ilm = torch.optim.Adam(ilm.parameters(), lr=1e-5)
    model_trainer = InsertLMTrainer(ilm, dev, optimize_ilm)

    # training
    #=============================================================#
    best_loss = model_trainer.evaluate(dev_loader)
    for epoch in range(3):
        logger.info(f"Training InsertLM -- Epoch {epoch}")
        model_trainer.train(train_loader)
        loss = model_trainer.evaluate(dev_loader)
        logger.info(f"Dev. Loss: {loss}")
        if loss < best_loss:
            logger.info(f"Update InsertLM dump {int(loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}")
            torch.save(ilm.state_dict(), f"../dump/ilm_{data}_finetune.pth")
            best_loss = loss
        logger.info("=" * 50)

elif mode == "test":
    # load sources
    #=============================================================#
    test_dataset= TemplateDataset(test_files, vb, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_transfer)

    # load model
    #=============================================================#
    ilm = InsertLM(len(vb))
    ilm.load_state_dict(torch.load(f"../dump/ilm_{data}_finetune.pth"))

    # construct trainer
    #=============================================================#
    model_trainer = InsertLMTrainer(ilm, dev, None)

    # generate permutation
    #=============================================================#
    model_trainer.transfer(test_loader, f"../out/ours/{data}_test_ilm.tsf", vb)
