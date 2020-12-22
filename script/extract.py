from fio import read, write

rev_keys = dict(enumerate(read("../data/gyafc/style.test.0")))
keys = {"".join(rev_keys[k].split()): k for k in rev_keys}
lines = read("../data/gyafc/style.test.sub")

print(f"total test samples: {len(lines)}")

# extract from PTO and CWSR
f1 = "../out/pto/GYAFC_pto.txt"
f2 = "../out/cwsr/GYAFC_cwsr.txt"

sample1 = read(f1)
sample2 = read(f2)

hit, idxs = [], []
for idx, l in enumerate(lines):
    if "".join(l.split()).strip() in keys:
        hit.append((idx, l))
        idxs.append(str(keys["".join(l.split()).strip()]))

write("../data/gyafc/cwsr_ver_idx.txt", idxs)
print(f"total hits: {len(hit)}")

# write to style file
tp1 = ["\t".join([l, sample1[idx], "1"]) for idx, l in hit]
tp2 = ["\t".join([l, sample2[idx], "1"]) for idx, l in hit]

write("../out/pto/gyafc_test.tsf", tp1)
write("../out/cwsr/gyafc_test.tsf", tp2)