from fio import read, write

model = "ours"

inp = f"../out/{model}/gyafc_test_ilm.tsf"
out = f"../out/{model}/gyafc_test_cwsr_ver.tsf"
idx_f = f"../data/gyafc/cwsr_ver_idx.txt"

lines = read(inp)
idxs = [int(idx) for idx in read(idx_f)]

lines = [lines[id_].strip() for id_ in idxs]

write(out, lines)