import torch

from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig


"""
python gram.py [hypothesis file]
"""

# hyp_file = sys.argv[1]
# assert os.path.exists(hyp_file)


dev = torch.device("cuda")

tkz = AutoTokenizer.from_pretrained("bert-base-uncased", mirror="tuna")

bert = BertForSequenceClassification(BertConfig()).to(dev)
bert.load_state_dict(torch.load("../dump/eval_disc.pth"))
"""
"""
