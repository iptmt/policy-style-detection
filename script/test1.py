import csv
from fio import read, write_csv

ds = "gyafc"

tp = {"train": "train", "dev": "val", "test": "test"}
lb = {"0": "P_0", "1": "P_9"}

# -------------------------------------------------
d = f"../data/{ds}/"

total_lines = [("txt", "style", "split")]
for t in tp:
    for l in lb:
        fn = d + ".".join(["style", t, l])
        lines = read(fn)
        lines = [(ln, lb[l], tp[t]) for ln in lines]
        total_lines += lines

write_csv("../tmp/gyafc.tsv", total_lines)
