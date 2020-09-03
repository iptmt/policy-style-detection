import os
import torch
import random
import numpy as np

from vocab import PAD_ID, BOS_ID, EOS_ID, PLH_ID, Vocab
from torch.utils.data import Dataset

from data_util import align_texts, noise_text_ids


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
        encode = lambda s: self.vocab.tokens_to_ids(s)[: self.max_len]
        # load
        sentences = [line.strip().split("\t") for line in f_obj]
        return [(encode(s.split()), encode(s_temp.split()), int(label)) for s, s_temp, label in sentences]
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def collate_fn_rank(batch_samples):
        sentences, temp_sentences, labels = zip(*batch_samples)
        aligned_sentences = torch.tensor(align_texts(sentences), dtype=torch.long)
        aligned_temps = torch.tensor(align_texts(temp_sentences), dtype=torch.long)
        ph_mask = (aligned_temps == PLH_ID).long()
        labels = torch.tensor(labels, dtype=torch.long)

        return aligned_sentences, aligned_temps, ph_mask, labels
    
    @staticmethod
    def collate_fn_transfer(batch_samples):
        sentences, temp_sentences, labels = zip(*batch_samples)
        sentences = [[BOS_ID] + s + [EOS_ID] for s in sentences]
        temp_sentences = [[BOS_ID] + s + [EOS_ID] for s in temp_sentences]

        temp_sentences = [list(filter(lambda x: x != PLH_ID, temp)) for temp in temp_sentences]
        aligned_sentences = torch.tensor(align_texts(sentences), dtype=torch.long)
        aligned_temps = torch.tensor(align_texts(temp_sentences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return aligned_sentences, aligned_temps, labels


class InsertLMDataset(Dataset):
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
        samples = []
        encode = lambda s: self.vocab.tokens_to_ids(s)[: self.max_len]
        for line in f_obj:
            text, order, label = line.strip().split("\t")
            tokens = [BOS_ID] + encode(text.strip().split()) + [EOS_ID]
            order = [0] + [int(od) for od in order.split()] + [0]
            label = int(label)
            samples += self.___segments(tokens, order, label)
        return samples
    
    def ___segments(self, tokens, order, label):
        t, p = np.array(tokens), np.array(order)
        samples = []
        for i in range((p > 0).sum() + 1):
            step = i + 1
            m1, m2 = (p < step), (p == step)
            out = m1 * t
            pre = 2 * m2 + m1 - 1
            if step > 1:
                m3 = (p == (step - 1))
                inp = m3 * PLH_ID + (1 - m3) * out
            else:
                inp = out
            inp = list(filter(lambda x: x > 0, inp))
            out = list(filter(lambda x: x > 0, out))
            pre = list(filter(lambda x: x != -1, pre))[:len(inp)]
            if sum(pre) == 0:
                pre[0] = 1
            assert len(inp) == len(out)
            assert len(inp) == len(pre)
            samples.append((inp, out, pre, label))
        return samples
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def collate_fn(batch_samples):
        inp, out, pos, label = zip(*batch_samples)
        inp = torch.tensor(align_texts(inp), dtype=torch.long)
        out = torch.tensor(align_texts(out), dtype=torch.long)
        pos = torch.tensor(align_texts(pos, pad_id=0), dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return inp, out, pos, label