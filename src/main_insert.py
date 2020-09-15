import os
import sys
import torch
import random
import numpy as np

from vocab import Vocab
from dataset import TemplateDataset
from torch.utils.data import DataLoader

from trainer import InsertLMTrainer
from nets.lm import InsertLM
from tool import create_logger
from data_util import mask_noise_file
from vocab import PLH

seed_num = 110
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
batch_size = 256
max_seq_len = None # no limit
dev = torch.device("cuda:0")
noise_p = 0.2

vocab_file = f"../dump/vocab_{data}.bin"
vb = Vocab.load(vocab_file)

data_dir = f"../data/{data}/"
tmp_dir = f"../tmp/"

# if mode == "pretrain":
#     # generate randomly masked samples
#     #=============================================================#
#     train_files = ["style.train.0", "style.train.1"]
#     dev_files = ["style.dev.0", "style.dev.1"]
    
#     ilm = InsertLM(len(vb))
#     optimize_ilm = torch.optim.Adam(ilm.parameters(), lr=1e-4)
#     model_trainer = InsertLMTrainer(ilm, dev, optimize_ilm)

#     for fn in dev_files:
#         f_info = fn.split(".")
#         mask_noise_file(data_dir + fn, tmp_dir + f"{fn}.{data}.mask", f_info[-1], noise_p)
#     dev_dataset = TemplateDataset([tmp_dir + f"{fn}.{data}.mask" for fn in dev_files], vb, max_len=max_seq_len)
#     dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_train)
#     best_loss = model_trainer.evaluate(dev_loader)

#     for epoch in range(epochs):
#         for fn in train_files:
#             f_info = fn.split(".")
#             mask_noise_file(data_dir + fn, tmp_dir + f"{fn}.{data}.mask", f_info[-1], noise_p)

#         # pretrain
#         #=============================================================#
#         train_dataset = TemplateDataset([tmp_dir + f"{fn}.{data}.mask" for fn in train_files], vb, max_len=max_seq_len)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TemplateDataset.collate_fn_train)

#         logger.info(f"Pre-training InsertLM -- Epoch {epoch}")
#         model_trainer.train(train_loader)
#         loss = model_trainer.evaluate(dev_loader)
#         logger.info(f"Dev. Loss: {loss}")
#         if loss < best_loss:
#             logger.info(f"Update InsertLM dump {int(loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}")
#             torch.save(ilm.state_dict(), f"../dump/ilm_{data}_pretrain.pth")
#             best_loss = loss
#         else:
#             break
#         logger.info("=" * 50)

#     for fn in train_files + dev_files:
#         os.remove(tmp_dir + f"{fn}.{data}.mask")

        
if mode == "train":
    train_files = [f"../tmp/{data}.train.mask"]
    dev_files = [f"../tmp/{data}.dev.mask"]
    # load sources
    #=============================================================#
    train_dataset = TemplateDataset(train_files, vb, max_len=max_seq_len)
    dev_dataset = TemplateDataset(dev_files, vb, max_len=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TemplateDataset.collate_fn_insert)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_insert)

    # create model
    #=============================================================#
    ilm = InsertLM(len(vb))
    # ilm.load_state_dict(torch.load(f"../dump/ilm_{data}_pretrain.pth"))

    # construct trainer
    #=============================================================#
    optimize_ilm = torch.optim.Adam(ilm.parameters(), lr=1e-4)
    model_trainer = InsertLMTrainer(ilm, dev, optimize_ilm)

    # training
    #=============================================================#
    best_loss = model_trainer.evaluate(dev_loader)
    for epoch in range(2):
        logger.info(f"Training InsertLM -- Epoch {epoch}")
        model_trainer.train(train_loader)
        loss = model_trainer.evaluate(dev_loader)
        logger.info(f"Dev. Loss: {loss}")
        if loss < best_loss:
            logger.info(f"Update InsertLM dump {int(loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}")
            torch.save(ilm.state_dict(), f"../dump/ilm_{data}.pth")
            best_loss = loss
        else:
            break
        logger.info("=" * 50)

elif mode == "inf":
    test_files = [f"../tmp/{data}.test.mask"]
    # load sources
    #=============================================================#
    test_dataset= TemplateDataset(test_files, vb, max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TemplateDataset.collate_fn_margin)

    # load model
    #=============================================================#
    ilm = InsertLM(len(vb))
    ilm.load_state_dict(torch.load(f"../dump/ilm_{data}.pth"))

    # construct trainer
    #=============================================================#
    model_trainer = InsertLMTrainer(ilm, dev, None)

    # generate permutation
    #=============================================================#
    model_trainer.inference(test_loader, f"../out/ours/{data}_test.tsf", vb)