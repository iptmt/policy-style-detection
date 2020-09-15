import os
import sys
import json
import torch
import random
import argparse
import numpy as np

from vocab import Vocab
from dataset import TemplateDataset
from torch.utils.data import DataLoader

from module.generator import InsertLM, MLM, RNN_S2S
from trainer import InsertLMTrainer, MLMTrainer, RNNTrainer
from tool import create_logger
from data_util import mask_noise_file
from vocab import PLH, BOS_ID

"""
example:
    python generate.py -d=yelp -m=train -t=rnn -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask
"""


parser = argparse.ArgumentParser(description="python generate.py -d=yelp -m=train -t=rnn\
     -trf=../tmp/yelp.train.mask -def=../tmp/yelp.dev.mask")
parser.add_argument('-dataset', '-d', type=str) # yelp / gyafc
parser.add_argument('-mode', '-m', type=str) # train / inf
parser.add_argument('-type', '-t', type=str) # rnn / rnn-attn / mlm / ilm

parser.add_argument('-train_file', '-trf', type=str, default="")
parser.add_argument('-dev_file', '-def', type=str, default="")
parser.add_argument('-test_file', '-tef', type=str, default="")

parser.add_argument('-max_length', '-l', type=int, default=24)
parser.add_argument('-epochs', '-ep', type=int, default=10)
parser.add_argument('-learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('-batch_size', '-bsz', type=int, default=256)
parser.add_argument('-device', '-dev', type=str, default='cuda')
parser.add_argument('-seed', type=int, default=110)

args = parser.parse_args()

# extended path
dump_dir = "../dump"
log_dir = "../log"
out_dir = "../out"

# create logger
logger = create_logger(log_dir, "_".join([args.type, args.dataset, args.mode] + ".log"))
logger.info(json.dumps(vars(args), indent=4))

# recap the parameters
args.device = torch.device(args.device)

# initialize random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load resources & priors
vb = Vocab.load(f"{dump_dir}/vocab_{args.dataset}.bin")

collate_fn_dict = {
    "rnn": {"train": TemplateDataset.collate_fn_margin, "inf": TemplateDataset.collate_fn_margin},
    "rnn-attn": {"train": TemplateDataset.collate_fn_margin, "inf": TemplateDataset.collate_fn_margin},
    "mlm": {"train": TemplateDataset.collate_fn_mask, "inf": TemplateDataset.collate_fn_mask},
    "ilm": {"train": TemplateDataset.collate_fn_insert, "inf": TemplateDataset.collate_fn_margin}
}

model_dict = {
    "rnn": (RNN_S2S(len(vb), args.max_length, BOS_ID, attention=False), RNNTrainer),
    "rnn-attn": (RNN_S2S(len(vb), args.max_length, BOS_ID, attention=True), RNNTrainer),
    "mlm": (MLM(len(vb)), MLMTrainer),
    "ilm": (InsertLM(len(vb)), InsertLMTrainer)
}

if args.mode == "train":
    # load dataset
    train_dataset = TemplateDataset([args.train_file], vb, max_len=args.max_length)
    dev_dataset = TemplateDataset([args.dev_file], vb, max_len=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_dict[args.type][args.mode])
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_dict[args.type][args.mode])

    # create model and trainer
    model, Trainer = model_dict[args.type]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, args.device, optimizer)

    # train
    best_loss = trainer.evaluate(dev_loader)
    for epoch in range(args.epochs):
        logger.info(f"Model Training @ Epoch {epoch}")
        trainer.train(train_loader)
        epoch_loss = trainer.evaluate(dev_loader)
        logger.info(f"Dev. Loss: {epoch_loss}")
        if epoch_loss < best_loss:
            logger.info(f"{int(epoch_loss*1e4)/1e4} <- {int(best_loss*1e4)/1e4}, dumping model...")
            torch.save(model.state_dict(), f"{dump_dir}/{'_'.join([args.type, args.dataset])}.pth")
            best_loss = epoch_loss
        else:
            break
        logger.info("=" * 50)

if args.mode == "inf":
    test_dataset= TemplateDataset([args.test_file], vb, max_len=args.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_dict[args.type][args.mode])

    model, Trainer = model_dict[args.type]
    # loading trained parameters
    model.load_state_dict(torch.load(f"{dump_dir}/{'_'.join([args.type, args.dataset])}.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, args.device, optimizer)

    trainer.inference(test_loader, f"{out_dir}/{args.ours}/yelp_test_{args.type}.tsf")