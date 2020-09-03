"""
python freq_ratio.py [corpus name] [`fit' or `inf'] [threshold](optional)

"""
import sys
sys.path.append("..")
import pickle
import numpy as np

from tqdm import tqdm
from nltk import everygrams, ngrams 

from vocab import Vocab, PLH_ID


ds = sys.argv[1]
md = sys.argv[2]
print(f"dataset: {ds}")
print(f"mode: {md}")
if md == "inf":
    ts = float(sys.argv[3])
    print(f"threshold: {ts}")

file_0 = f"../../data/{ds}/style.train.0"
file_1 = f"../../data/{ds}/style.train.1"
test_0 = f"../../data/{ds}/style.test.0"
test_1 = f"../../data/{ds}/style.test.1"
gram_file = f"../../tmp/grams_{ds}.bin"
vocab_file = f"../../dump/vocab_{ds}.bin"
output_file = f"../../tmp/{ds}.test.mask.ngrams"

vocab = Vocab.load(vocab_file)

def cal_freq_ngrams(word_lv_sents):
    gram_freq = dict()
    with tqdm(total=len(word_lv_sents)) as bar:
        for sent in word_lv_sents:
            max_l = min([len(sent), 4])
            for gram in everygrams(sent, max_len=max_l):
                if gram not in gram_freq:
                    gram_freq[gram] = 1
                else:
                    gram_freq[gram] += 1
            bar.update()
    return gram_freq

def mask_sentences(word_lv_sents, src_dict, tgt_dict, threshold, v_smooth=1.0, mask_token=PLH_ID):
    masked_sents = []
    with tqdm(total=len(word_lv_sents)) as bar:
        for sent in word_lv_sents:
            sent = np.array(sent)
            inds = np.zeros(len(sent), dtype=np.int)
            for idx in range(len(sent)):
                for i in range(1, 5):
                    if idx + i <= len(sent):
                        gram = tuple(sent[idx: idx + i])
                        src_score = src_dict[gram] if gram in src_dict else 0
                        tgt_score = tgt_dict[gram] if gram in tgt_dict else 0
                        salience = (src_score + v_smooth) / (tgt_score + v_smooth)
                        if salience >= threshold:
                            inds[idx: idx + i] = 1
            masked_sent = sent * (1 - inds) + PLH_ID * inds
            masked_sents.append(masked_sent.tolist())
            bar.update()
    return masked_sents


if md == "fit":
    corpus_0 = [vocab.tokens_to_ids(line.split()) for line in open(file_0, "r", encoding="utf-8")]
    corpus_1 = [vocab.tokens_to_ids(line.split()) for line in open(file_1, "r", encoding="utf-8")]
    grams_0 = cal_freq_ngrams(corpus_0)
    grams_1 = cal_freq_ngrams(corpus_1)
    pickle.dump([grams_0, grams_1], open(gram_file, "wb"), pickle.HIGHEST_PROTOCOL)

if md == "inf":
    grams_0, grams_1 = pickle.load(open(gram_file, "rb"))
    corpus_0 = [vocab.tokens_to_ids(line.split()) for line in open(test_0, "r", encoding="utf-8")]
    corpus_1 = [vocab.tokens_to_ids(line.split()) for line in open(test_1, "r", encoding="utf-8")]
    masked_0 = mask_sentences(corpus_0, grams_0, grams_1, ts)
    masked_1 = mask_sentences(corpus_1, grams_1, grams_0, ts)
    labels_0 = ["0" for _ in range(len(corpus_0))]
    labels_1 = ["1" for _ in range(len(corpus_1))]

    with open(output_file, "w+", encoding='utf-8') as f:
        for obj in (zip(corpus_0, masked_0, labels_0), zip(corpus_1, masked_1, labels_1)):
            for src, masked, label in obj:
                src = " ".join(vocab.ids_to_tokens(src))
                masked = " ".join(vocab.ids_to_tokens(masked))
                f.write(src + "\t" + masked + "\t" + label + "\n")