import re, sys
from fio import read, write

ds = sys.argv[1]
def replace(text):
    return re.sub(r"\[.*?\]", "<PH>", text)

tmp = "/home/zaracs/workspace/policy-style-detection/tmp/"
data = f"/home/zaracs/workspace/policy-style-detection/data/{ds}/"
base = "/home/zaracs/workspace/baselines/tagger-generator/data/"

f0_o = "style.test.0"
f1_o = "style.test.1"

f0 = "entagged_parallel.test.en.P_0"
f1 = "entagged_parallel.test.en.P_9"

f0_ = "entagged_parallel.test.tagged.P_0"
f1_ = "entagged_parallel.test.tagged.P_9"

t0 = read(data + f0_o)
t1 = read(data + f1_o)
t0_ = set(read(base + f0))
t1_ = set(read(base + f1))

tt0 = [x + "\t" + replace(x_) + "\t" + "0" for x, x_ in zip(read(base + f0), read(base + f0_))]
tt1 = [x + "\t" + replace(x_) + "\t" + "1" for x, x_ in zip(read(base + f1), read(base + f1_))]

for l in t0:
    if l not in t0_:
        tt0.append(l + "\t" + l + "\t" + "0")
for l in t1:
    if l not in t1_:
        tt1.append(l + "\t" + l + "\t" + "1")

if ds == "yelp":
    assert len(tt0) == 500
    assert len(tt1) == 500
elif ds == "gyafc":
    assert len(tt0) == 1332

write(tmp + f"{ds}.test.mask.tfidf", tt0 + tt1)
