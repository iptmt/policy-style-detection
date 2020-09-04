import os
import torch
import random
import numpy as np

from vocab import PAD_ID, BOS_ID, EOS_ID, PLH_ID, Vocab
from torch.utils.data import Dataset

from data_util import align_texts, noise_text_ids, iter_samples


class StyleDataset(Dataset):
    def __init__(self, files, vocab, max_len=None):
        super().__init__()
        self.files = files
        self.vocab = vocab
        self.max_len = max_len
        self.samples = self._load()
    
    def _load(self):
        samples = []
        for file in self.files:
            assert os.path.exists(file)
            label = int(file.split(".")[-1])
            f_obj = open(file, 'r', encoding='utf-8')
            samples += self.__load_lines(f_obj, label)
            f_obj.close()
        return samples
    
    def __load_lines(self, f_obj, label):
        encode = lambda s: self.vocab.tokens_to_ids(s)[: self.max_len]
        # load
        sentences = [line.strip().split() for line in f_obj]
        # filter null sentence
        sentences = list(filter(lambda s: s, sentences))
        return [(encode(s), label) for s in sentences]
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def collate_fn(batch_samples):
        sentences, labels = zip(*batch_samples)
        sent_tensors = torch.tensor(align_texts(sentences, pad_id=PAD_ID), dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return sent_tensors, label_tensor, (sent_tensors > 0).long()
    
    @staticmethod
    def collate_fn_noise(p, noise_id, noise_type): # noise_type = "mask" or "insert"
        def fn(batch_samples):
            sentences, labels = zip(*batch_samples)
            aligned_sentences, noised_sentences = noise_text_ids(sentences, p, noise_id, noise_type)
            return (
                torch.tensor(aligned_sentences, dtype=torch.long),
                torch.tensor(noised_sentences, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            )
        return fn


class TemplateDataset(Dataset):
    def __init__(self, files, vocab, max_len=None):
        super().__init__()
        self.files = files
        self.vocab = vocab
        self.max_len = max_len
        self.samples = self._load()
    
    def _load(self):
        samples = []
        for file in self.files:
            assert os.path.exists(file)
            f_obj = open(file, 'r', encoding='utf-8')
            samples += self.__load_lines(f_obj)
            f_obj.close()
        return samples
    
    def __load_lines(self, f_obj):
        encode = lambda s: [BOS_ID] + self.vocab.tokens_to_ids(s)[: self.max_len] + [EOS_ID]
        # load
        sentences = [line.strip().split("\t") for line in f_obj]
        return [(encode(s.split()), encode(s_temp.split()), int(label)) for s, s_temp, label in sentences]
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def collate_fn_train(batch_samples):
        masked_ids, ids, removed_ids, inds, label_mlm, label_slot = [], [], [], [], [], []
        for s, ns, l in  batch_samples:
            ms, s, rs, i = iter_samples(s, ns)
            masked_ids += ms
            ids += s
            label_mlm += [l] * len(ms)
            removed_ids += rs
            inds += i
            label_slot += [l] * len(rs)
        if masked_ids:
            alg_masked_ids = torch.tensor(align_texts(masked_ids), dtype=torch.long)
            alg_ids = torch.tensor(align_texts(ids), dtype=torch.long)
            label_mlm = torch.tensor(label_mlm, dtype=torch.long)
        else:
            alg_masked_ids, alg_ids, label_mlm = None, None, None
        alg_removed_ids = torch.tensor(align_texts(removed_ids), dtype=torch.long)
        alg_inds = torch.tensor(align_texts(inds, pad_id=-1), dtype=torch.long)
        label_slot = torch.tensor(label_slot, dtype=torch.long)

        return alg_masked_ids, alg_ids, label_mlm, alg_removed_ids, alg_inds, label_slot
    
    @staticmethod
    def collate_fn_inf(batch_samples):
        sentences, temp_sentences, labels = zip(*batch_samples)
        temp_sentences = [list(filter(lambda x: x != PLH_ID, temp)) for temp in temp_sentences]

        aligned_sentences = torch.tensor(align_texts(sentences), dtype=torch.long)
        aligned_temps = torch.tensor(align_texts(temp_sentences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return aligned_sentences, aligned_temps, labels



if __name__ == "__main__":
    import time
    from data_util import mask_noise_file
    from vocab import Vocab
    from torch.utils.data import DataLoader

    file = "../data/yelp/style.train.0"
    outf = "../tmp/test.0"
    mask_noise_file(file, outf, 1, 0.15)

    vocab = Vocab.load("../dump/vocab_yelp.bin")

    dataset = TemplateDataset([outf], vocab, None)
    loader = DataLoader(dataset, 512, False, collate_fn=TemplateDataset.collate_fn_train)
    cnt = 1
    for a, b, c, d, e, f in loader:
        if a is not None:
            cnt += len(a)
        print(cnt)
        #     print(a)
        #     print(b)
        #     print(c)
        # print(d)
        # print(e)
        # print(f)
        # print("=" * 100)
        # time.sleep(2)
        # print(cnt)