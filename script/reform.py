import os
from fio import read, write


baseline = "pto"
dataset = "yelp"

# ori_0_p = f"../data/{dataset}/style.test.0"
# ori_1_p = f"../data/{dataset}/style.test.1"

# hyp_0_p = f"../out/{baseline}/{dataset}.test.0.hyp"
# hyp_1_p = f"../out/{baseline}/{dataset}.test.1.hyp"

# if os.path.exists(hyp_0_p) and os.path.exists(hyp_1_p):
#     ori_0, ori_1, hyp_0, hyp_1 = read(ori_0_p), read(ori_1_p), read(hyp_0_p), read(hyp_1_p)

#     objs = []
#     for label, pairs in enumerate([zip(ori_0, hyp_0), zip(ori_1, hyp_1)]):
#         for ori, hyp in pairs:
#             objs.append(ori + "\t" + hyp + "\t" + str(1 - label))

#     write(f"../out/{baseline}/{dataset}_test.tsf", objs)

# files = os.listdir(f"../out/{baseline}/")
# for file in files:
#     if dataset in file and not file.endswith(".tsf"):
#         os.remove(f"../out/{baseline}/{file}")


file = f"../out/{baseline}/{dataset}_test.tsf"

lines = read(file)

new_lines = []
for l in lines:
    ori, tsf, label = l.split("\t")
    label = str(1 - int(label))
    new_lines.append("\t".join([ori, tsf, label]))

write(file, new_lines)