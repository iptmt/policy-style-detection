import random
import numpy as np
from vocab import PLH_ID, PLH, PAD_ID


def align_texts(texts, pad_id=PAD_ID, pad_mask=False):
    max_len = max([len(text) for text in texts])
    if pad_mask:
        mask = [[1] * len(text) + [0] * (max_len - len(text)) for text in texts]
    texts = [text + [pad_id] * (max_len - len(text)) for text in texts]
    if pad_mask:
        return texts, mask
    else:
        return texts


def noise_text_ids(texts, p, noise_id, noise_type):
    pad_texts = align_texts(texts)
    noise_texts = [noise_text_ids_(text, p, noise_id, noise_type) for text in texts]
    noise_texts = align_texts(noise_texts)
    return pad_texts, noise_texts


def noise_text(text, p, mask_token):
    text = np.array(text)
    up = np.random.uniform(size=len(text))
    text[up < p] = mask_token
    return text.tolist()

def noise_text_ids_(text, p, noise_id, noise_type):
    new_text = []
    pos = np.random.uniform(size=len(text))
    for word, v in zip(text, pos):
        if v < p:
            new_text.append(noise_id)
            if noise_type == "mask":
                continue
            p2 = random.random()
            if 0.7 < p2 <= 0.9:
                new_text += [noise_id]
            elif p2 > 0.9:
                new_text += [noise_id] * 2
            if noise_type == "insert":
                new_text.append(word)
        else:
            new_text.append(word)
    return new_text