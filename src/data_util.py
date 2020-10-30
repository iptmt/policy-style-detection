import math
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

# ===============================================================

def mask_noise_file(input_file, output_file, label, p):
    lines = [line.strip() for line in open(input_file, 'r')]
    noised_lines = [" ".join(mask_noise_text(line.split(), p, PLH)) for line in lines]
    with open(output_file, 'w+') as f:
        for l, nl in zip(lines, noised_lines):
            f.write('\t'.join([l, nl, str(label)]) + "\n")

def mask_noise_text(text, p, mask_token):
    text = np.array(text)
    up = np.random.uniform(size=len(text))
    text[up < p] = mask_token
    return text.tolist()

def iter_samples(origin, ids):
    ms, s, ms_filter, idt = [], [], [], []
    origin, ids = np.array(origin), np.array(ids)
    while True:
        inds = (ids == PLH_ID).astype(np.int)
        mask = ((inds - (np.concatenate([inds[-1:], inds[:-1]]))) > 0).astype(np.int)
        if mask.sum() == 0:
            ms_filter.append(origin.tolist())
            idt.append([0] * (origin.size - 1) + [1])
            break
        else:
            a, b = filter_right_PLH_id(origin, ids)
            c, d = remove_PLH_id(a)
            ms.append(a)
            s.append(b)
            ms_filter.append(c)
            idt.append(d)
        ids = (mask * origin) + (1 - mask) * ids
    return ms, s, ms_filter, idt


def filter_right_PLH_id(origin, ids):
    new_ids, output, flag = [], [], 0
    for oid, id_ in zip(origin, ids):
        if id_ != PLH_ID:
            new_ids.append(id_)
            output.append(oid)
            flag = 0
        else:
            if flag == 0:
                new_ids.append(id_)
                output.append(oid)
                flag = 1
    return new_ids, output

def remove_PLH_id(ids):
    new_ids, inds, flag = [], [], 0
    for id_ in ids[::-1]:
        if id_ == PLH_ID:
            flag = 1
        else:
            new_ids.insert(0, id_,)
            if flag == 1:
                inds.insert(0, 1)
                flag = 0
            else:
                inds.insert(0, 0)
    return new_ids, inds


# ===============================================================

def noise_text_ids(texts, p, noise_id, noise_type):
    pad_texts = align_texts(texts)
    noise_texts = [noise_text_ids_(text, p, noise_id, noise_type) for text in texts]
    noise_texts = align_texts(noise_texts)
    return pad_texts, noise_texts

def noise_text_ids_(text, p, noise_id, noise_type):
    if noise_type == "mask":
        return mask_noise_ids(text, noise_id, p)
    elif noise_type == "insert":
        return insert_noise_ids(text, noise_id, p)
    elif noise_type == "random":
        return random_noise_ids(text, noise_id, p)
    else:
        raise ValueError

def mask_noise_ids(text, noise_id, p=0.15):
    inds = np.random.uniform(size=len(text))
    # text = np.array(text, dtype=np.long)
    # for idx, i in enumerate(inds):
    #     if i < p:
    #         text[idx - 1: idx + math.ceil(random.random() * 3) - 1] = noise_id
    # return text.tolist()
    return list(map(lambda x: x[0] if x[1] > p else noise_id, zip(text, inds)))

def insert_noise_ids(text, noise_id, p=0.15):
    inds = np.random.uniform(size=len(text))
    new_text = []
    for x, i in zip(text, inds):
        if i < p:
            r = random.random()
            if r <= 0.7:
                new_text += [noise_id]
            elif 0.7 < r <= 0.9:
                new_text += [noise_id] * 2
            else:
                new_text += [noise_id] * 3
        new_text.append(x)
    return new_text

def random_noise_ids(text, noise_id, p=0.15):
    text = mask_noise_ids(text, noise_id, p)
    text = list(filter(lambda x: x != noise_id, text))
    return insert_noise_ids(text ,noise_id, p)




if __name__ == "__main__":
    a = [100, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100]
    inds = np.random.uniform(size=len(a))
    b = random_noise_ids(a, 0,  0.3)
    print(b)
