import os
import sys
from collections import defaultdict
# from bert_score import BERTScorer
from nltk.translate.bleu_score import corpus_bleu

"""
python cp.py [corpus name] [hypothesis file]
"""

ds = sys.argv[1]
hyp_file = sys.argv[2]

assert os.path.exists(hyp_file)

ref_path = f"../data/{ds}/human_ref/"

# load hypothesis
cands = [line.strip().split("\t")[1] for line in open(hyp_file, 'r')]

# prepare reference list
ref_files = os.listdir(ref_path)
ref_dict = defaultdict(list)
for name in ref_files:
    _, _, sty, index = name.split(".")
    ref_dict[index].append((sty, ref_path + name))
for index in ref_dict:
    ref_dict[index].sort(key=lambda x: x[0])
    refs_i = []
    for _, file_path in ref_dict[index]:
        refs_i += [line.strip() for line in open(file_path, 'r')]
    ref_dict[index] = refs_i
ref_list = [refs for refs in ref_dict.values()]
ref_sents = [ref for refs in ref_list for ref in refs]
ref_list = list(zip(*ref_list))

cands = [c.strip().split() for c in cands]
ref_list = [[ref.strip().split() for ref in refs] for refs in ref_list]

bleu = corpus_bleu(ref_list, cands)

print("BLEU: %.4f" % bleu)
# scorer = BERTScorer(model_type="albert-xlarge-v2", lang="en", rescale_with_baseline=True, idf=True, idf_sents=ref_sents, batch_size=32)

# P, R, F1 = scorer.score(cands, ref_list)
# P, R, F1 = P.mean().item(), R.mean().item(), F1.mean().item()

# print("P: %.4f; R: %.4f; F1: %.4f." % (P, R, F1))