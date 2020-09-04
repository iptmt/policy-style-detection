import random, time
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
    noised_lines = [" ".join(mask_noise_text(line, p, PLH)) for line in lines]
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
        inds = (ids == PAD_ID).astype(np.int) #TODO:
        mask = ((inds - (np.concatenate([inds[-1:], inds[:-1]]))) > 0).astype(np.int)
        print(mask.sum())
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
        time.sleep(0.5)
        ids = (mask * origin) + (1 - inds) * ids
    return ms, s, ms_filter, idt
    

def filter_right_PLH_id(origin, ids):
    new_ids, output, flag = [], [], 0
    for oid, id_ in zip(origin, ids):
        if id_ != PAD_ID: #TODO:
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
        if id_ == PAD_ID: #TODO:
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


if __name__ == "__main__":
    a = [100, 1, 2, 3, 4, 100]
    b = [100, 0, 0, 1, 0, 100]
    print(a)
    print(b)
    print('=' * 100)
    a, b, c, d = iter_samples(a, b)
    print(a)
    print(b)
    print(c)
    print(d)