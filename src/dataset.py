import os
import torch
import random
import numpy as np

from vocab import PAD_ID, BOS_ID, EOS_ID, PLH_ID, Vocab
from torch.utils.data import Dataset

from data_util import align_texts, noise_text_ids, filter_right_PLH_id, remove_PLH_id


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
        for masked_ids, ids, removed_ids, inds = [], [], [], []
        for s, ns, l in  zip(*batch_samples):
            ms, s = filter_right_PLH_id(s, ns)
            fs, i = remove_PLH_id(ms)

        # aligned_sentences = torch.tensor(align_texts(sentences), dtype=torch.long)
        # aligned_temps = torch.tensor(align_texts(temp_sentences), dtype=torch.long)
        # ph_mask = (aligned_temps == PLH_ID).long()
        # labels = torch.tensor(labels, dtype=torch.long)

        return aligned_sentences, aligned_temps, ph_mask, labels
    
    @staticmethod
    def collate_fn_inf(batch_samples):
        sentences, temp_sentences, labels = zip(*batch_samples)
        sentences = [[BOS_ID] + s + [EOS_ID] for s in sentences]
        temp_sentences = [[BOS_ID] + s + [EOS_ID] for s in temp_sentences]

        temp_sentences = [list(filter(lambda x: x != PLH_ID, temp)) for temp in temp_sentences]
        aligned_sentences = torch.tensor(align_texts(sentences), dtype=torch.long)
        aligned_temps = torch.tensor(align_texts(temp_sentences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return aligned_sentences, aligned_temps, labels