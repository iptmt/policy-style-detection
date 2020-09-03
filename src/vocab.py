import os
import pickle
from nltk.tokenize import word_tokenize

from collections import Counter

PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PLH = "<PH>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
PLH_ID = 4

class Vocab:
    def __init__(self, file_list, min_freq):
        self.file_list = file_list
        self.min_freq = min_freq

        self.word_dict = self._read(file_list)
        self.rev_dict = {v: k for k, v in self.word_dict.items()}

        self.special_tokens = {PAD_ID, BOS_ID, EOS_ID, PLH_ID}

    def _read(self, file_list):
        words_list = []
        for file in file_list:
            with open(file, 'r', encoding="utf-8") as rf:
                for line in rf:
                    words_list += line.strip().lower().split()
        freq_dec_words = Counter(words_list).most_common()
        word_dict = {
            PAD: PAD_ID, BOS: BOS_ID, EOS: EOS_ID, UNK: UNK_ID, PLH: PLH_ID
        }
        for word, freq in freq_dec_words:
            if freq >= self.min_freq:
                word_dict[word] = len(word_dict)
        return word_dict
    
    def tokens_to_ids(self, tokens):
        return [self.word_dict[t] if t in self.word_dict else UNK_ID for t in tokens]
    
    def ids_to_tokens(self, ids):
        return [self.rev_dict[id_] for id_ in ids]

    def tensor_to_template(self, tensor, pad_mask=None):
        ids = tensor.cpu().numpy().tolist()
        if pad_mask is not None:
            assert tensor.size(0) == pad_mask.size(0)
            ids = ids[:pad_mask.sum(0).item()]
        ids = map(lambda x: x if x != PAD_ID else PLH_ID, ids)
        return " ".join(self.ids_to_tokens(ids))

    def tensor_to_sent(self, tensor):
        ids = tensor.cpu().numpy().tolist()
        ids = filter(lambda t: t not in self.special_tokens, ids)
        return self.ids_to_tokens(ids)
    
    def save(self, path):
        assert not os.path.exists(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path):
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __len__(self):
        return len(self.word_dict)
    
    def __getitem__(self, x):
        if isinstance(x, int):
            return self.rev_dict[x]
        elif isinstance(x, str):
            return self.word_dict[x] if x in self.word_dict else UNK_ID
        else:
            raise ValueError


if __name__ == "__main__":
    import sys

    """
    python vocab.py [corpus_name] [min_frequency]
    """
    data = sys.argv[1]
    freq = sys.argv[2]
    files = [f"../data/{data}/style.train.0", f"../data/{data}/style.train.1"]
    vocab = Vocab(files, int(freq))
    print(f"dataset = {data}")
    print(f"vocab size = {len(vocab)}")
    # save & load
    vocab.save(f"../dump/vocab_{data}.bin")
    v2 = Vocab.load(f"../dump/vocab_{data}.bin")